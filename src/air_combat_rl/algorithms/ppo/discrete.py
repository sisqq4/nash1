"""Masked discrete PPO components for the 29-action blue escape action set."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from air_combat_rl.algorithms.ppo.loss import clipped_surrogate_loss
from air_combat_rl.algorithms.ppo.rollout_buffer import RolloutBuffer

LOG_2PI = float(np.log(2.0 * np.pi))

def masked_logits(logits, mask):
    m=np.asarray(mask,bool); return np.where(m, logits, -1e9)
def softmax(logits):
    z=np.asarray(logits,float)-np.max(logits,axis=-1,keepdims=True); e=np.exp(z); return e/np.sum(e,axis=-1,keepdims=True)

@dataclass(slots=True)
class DiscreteSample:
    action_id:int; log_prob:float; entropy:float; value:float; logits:np.ndarray

class PPODiscreteActorCritic:
    def __init__(self, obs_dim:int, action_dim:int=29, seed:int|None=None):
        self.obs_dim=obs_dim; self.action_dim=action_dim; self.rng=np.random.default_rng(seed)
        self.logit_w=self.rng.normal(0,0.01,(obs_dim,action_dim)); self.logit_b=np.zeros(action_dim)
        self.value_w=self.rng.normal(0,0.01,(obs_dim,)); self.value_b=np.zeros(1)
    def forward(self, obs):
        x=np.asarray(obs,float); single=x.ndim==1
        if single: x=x.reshape(1,-1)
        logits=x@self.logit_w+self.logit_b; values=x@self.value_w+self.value_b[0]
        return (logits[0], float(values[0])) if single else (logits, values)
    def act(self, obs, action_mask, deterministic=False):
        logits,value=self.forward(obs); ml=masked_logits(logits, action_mask); probs=softmax(ml.reshape(1,-1))[0]
        action=int(np.argmax(probs)) if deterministic else int(self.rng.choice(self.action_dim,p=probs))
        lp=float(np.log(max(probs[action],1e-12))); ent=float(-np.sum(probs*np.log(np.clip(probs,1e-12,None))))
        return DiscreteSample(action,lp,ent,value,logits)
    def evaluate_actions(self, observations, actions, masks):
        logits, values=self.forward(observations); probs=softmax(masked_logits(logits,masks)); idx=np.asarray(actions,int)
        lp=np.log(np.clip(probs[np.arange(len(idx)),idx],1e-12,None)); ent=-np.sum(probs*np.log(np.clip(probs,1e-12,None)),axis=1)
        return lp, ent, values
    def trainable_vectors(self): return [self.logit_b,self.value_b]
    def state_dict(self): return {"logit_w":self.logit_w,"logit_b":self.logit_b,"value_w":self.value_w,"value_b":self.value_b,"obs_dim":self.obs_dim,"action_dim":self.action_dim,"rng_state":self.rng.bit_generator.state}

@dataclass(frozen=True, slots=True)
class PPODiscreteTrainerConfig:
    rollout_steps:int=32; gamma:float=0.99; gae_lambda:float=0.95; learning_rate:float=0.003; clip_range:float=0.2; vf_coef:float=0.5; ent_coef:float=0.01; max_grad_norm:float=0.5; gradient_epsilon:float=1e-3; bootstrap_truncated:bool=True

class DiscretePPOPolicy:
    algorithm_name="ppo_discrete"
    def __init__(self, actor_critic, env): self.actor_critic=actor_critic; self.env=env
    def act(self, observation, deterministic=False):
        mask=self.env.actions.action_mask(self.env.platform); s=self.actor_critic.act(observation,mask,deterministic); cmd=self.env.actions.command_for(s.action_id,self.env.platform)
        from air_combat_rl.interfaces.policy import PolicyDecision
        return PolicyDecision(self.algorithm_name, s.logits, None, None, s.action_id, cmd, 0.0, s.value, s.log_prob)

class PPODiscreteTrainer:
    def __init__(self, env, actor_critic, config=None): self.env=env; self.actor_critic=actor_critic; self.config=config or PPODiscreteTrainerConfig(); self.buffer=RolloutBuffer(); self.global_step=0; self._last_metrics={}; self.completed_outcomes=[]
    def collect_rollout(self, reset_seed=None):
        self.buffer.clear(); obs,info=self.env.reset(reset_seed); ep_start=True; masks=[]; next_obs=[]
        for _ in range(self.config.rollout_steps):
            mask=info["action_mask"]; s=self.actor_critic.act(obs,mask); r=self.env.step(s.action_id)
            self.buffer.add(obs,s.action_id,r.reward,s.value,s.log_prob,r.terminated,r.truncated,ep_start,executed_action_id=s.action_id,action_mask=mask,next_observation=r.observation)
            obs=r.observation; info=r.info; ep_start=r.terminated or r.truncated; self.global_step+=1
            if ep_start: self.completed_outcomes.append(info.get("outcome")); obs,info=self.env.reset()
        last_value=self.actor_critic.act(obs,info["action_mask"],True).value
        self.buffer.compute_returns_and_advantages(last_value,self.config.gamma,self.config.gae_lambda,bootstrap_truncated=self.config.bootstrap_truncated); return self.buffer
    def _objective(self):
        lp,ent,vals=self.actor_critic.evaluate_actions(np.asarray(self.buffer.observations,float),np.asarray(self.buffer.actions,int),np.asarray(self.buffer.action_masks,bool))
        return clipped_surrogate_loss(self.buffer.log_probs,lp,self.buffer.advantages,vals,self.buffer.returns,ent,self.config.clip_range,self.config.vf_coef,self.config.ent_coef)
    def train_one_update(self):
        if not self.buffer.observations: self.collect_rollout()
        params=self.actor_critic.trainable_vectors(); grads=[]
        for param in params:
            grad=np.zeros_like(param); it=np.nditer(param,flags=["multi_index"],op_flags=["readwrite"])
            for _ in it:
                idx=it.multi_index; orig=float(param[idx]); eps=self.config.gradient_epsilon; param[idx]=orig+eps; plus=self._objective().total_loss; param[idx]=orig-eps; minus=self._objective().total_loss; param[idx]=orig; grad[idx]=(plus-minus)/(2*eps)
            grads.append(grad)
        norm=float(np.sqrt(sum(np.sum(g*g) for g in grads))); scale=min(1.0,self.config.max_grad_norm/(norm+1e-8))
        for p,g in zip(params,grads): p-=self.config.learning_rate*scale*g
        st=self._objective(); hist={int(a):self.buffer.executed_action_ids.count(a) for a in sorted(set(self.buffer.executed_action_ids))}
        self._last_metrics={"global_step":self.global_step,"policy_loss":st.policy_loss,"value_loss":st.value_loss,"entropy":-st.entropy_loss,"approx_kl":st.approx_kl,"clip_fraction":st.clip_fraction,"gradient_norm":norm,"projected_action_histogram":hist}
        return self._last_metrics
    def get_training_metrics(self): return dict(self._last_metrics)
