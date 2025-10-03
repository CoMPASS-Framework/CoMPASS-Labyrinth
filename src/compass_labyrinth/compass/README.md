# Level 1 & Level 2 

Two Levels of the hierarchical framework to decode latent behavioral states from complex spatial navigation data in a naturalistic maze. Level 1 captures both fine-grained motor dynamics and Level 2 captures higher-order cognitive strategies or states using unsupervised probabilistic models such as Bayesian Gaussian Mixture Models and Gaussian Mixture Models-Hidden Markov Models.

---

## Level 1 

### Goal: 
- Identify moment-to-moment behavioral states based on raw or smoothened movement features.

### Modeling Strategy:
- Apply Hidden Markov Model (HMM) (with Gaussian emissions) to short-timescale features.
- Features/Data Streams:
  -	 Step size
  -	 Turn angle
  -	 (Optionally) Smoothed or Log-Transformed variants

### Output:
- Hidden States decoded:
  -	 State 1: Low step length + High turn angle → Surveillance
  -	 State 2: High step length + Low turn angle → Ambulation
- Per-frame latent state sequence 

----

## Level 2  

### Goal: 
- Identify internal goal states that guide transitions across decision points in the maze, in pursuit of the target zone.

### Modeling Strategy:
- Use the Level 1 state sequence as input
- Combine with reward-contextual features:
  -  Sternum-based angular deviation (from path leading to the reward path)
  -  Value-based Distance to target
  -  KDE-based proximity to target zone 
- Fit a Bayesian Gaussian Mixture Model (BGMM) to extract latent states (clusters)
- Feed BGMM outputs into a GMM-HMM to model longer timescale dependencies across time and maze structure

### Output:
- States decoded:
  -	 Oriented
  -	 Non-Oriented   
- Per-frame latent state sequence 


## Level 1 & Level 2 Composite States
- Ambulatory, Oriented
- Ambulatory, Non-Oriented
- Active Surveillance, Oriented
- Active Surveillance, Non-Oriented

