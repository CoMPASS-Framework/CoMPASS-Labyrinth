# CoMPASS-Labyrinth

CoMPASS-Labyrinth is a unified computational and behavioral framework for analyzing goal-directed navigation in complex, ethologically valid maze environments using hierarchical probabilistic models. This project integrates behavioral modeling with neural data analysis to uncover latent cognitive states and their underlying neural dynamics during complex decision-making tasks.

<p align="center">
  <img src="https://raw.githubusercontent.com/CoMPASS-Framework/CoMPASS-Labyrinth/main/media/compass_logo.png" alt="CoMPASS Logo" width="220"/>
  &nbsp;&nbsp;&nbsp;
  <a href="https://raw.githubusercontent.com/CoMPASS-Framework/CoMPASS-Labyrinth/main/media/labyrinth_demo.mp4">
    <img src="https://raw.githubusercontent.com/CoMPASS-Framework/CoMPASS-Labyrinth/main/media/labyrinth_thumbnail.png" alt="Watch Demo" width="220"/>
  </a>
  &nbsp;&nbsp;&nbsp;
  <img src="https://raw.githubusercontent.com/CoMPASS-Framework/CoMPASS-Labyrinth/main/media/maze_layout.png" alt="Maze Layout" width="220"/>
</p>

---

<p align="center">
  <img src="https://raw.githubusercontent.com/CoMPASS-Framework/CoMPASS-Labyrinth/main/media/compass_framework.png" alt="Framework" width="900"/>
</p>

## Key Features

1. **Naturalistic Maze Framework**: A Novel Labyrinth maze paradigm that elicits spontaneous, intrinsically motivated, and untrained navigation behavior in rodents, closely mimicking real-world foraging.

2. **CoMPASS**: A Hierarchical Probabilistic Framework integrating local movement dynamics with goal-directed cognitive states.

3. **Latent State Inference**: Identification of fine-grained cognitive states that underlie navigation strategies, beyond what is captured by task performance alone.

4. **Neural-Behavioral Integration**: Linking probabilistically inferred behavioral states with neural oscillatory signatures, reflecting how internal cognitive processes manifest in circuit-level dynamics.

5. **Translational Relevance**: Sensitive detection of early cognitive deficits in models of neurodegenerative disease (e.g., App-KI mice), with broader implications for human cognition, learning, and memory.

## Workflow Overview

<p align="center">
  <img src="https://raw.githubusercontent.com/CoMPASS-Framework/CoMPASS-Labyrinth/refs/heads/main/media/compass_workflow.png" alt="Workflow" width="900"/>
</p>

The CoMPASS-Labyrinth workflow includes:

1. **Data Preprocessing** - DLC pose estimation and grid processing
2. **Task Performance Analysis** - Behavioral metrics and success rates
3. **Simulated Agent Modeling** - Computational models of behavior
4. **CoMPASS Level 1** - Fine-grained motor state inference (HMM)
5. **CoMPASS Level 2** - Goal-directed cognitive state inference (BGMM + GMM-HMM)
6. **Post-hoc Analysis** - Spatial, temporal, and bout-wise analysis

## Citation

If you use this framework, please cite our manuscript:

!!! info "Citation"
    **Decoding hidden goal-directed navigational states and their neuronal representations using a novel labyrinth paradigm and probabilistic modeling framework. bioarxiv (2025)**

    Patrick S Honma, Shreya C Bangera*, Reuben R Thomas, Nicholas Kaliss, Dan Xia, Jorge J Palop

    DOI: <https://doi.org/10.1101/2025.11.13.688348>

## License

This project is licensed under the *GNU General Public License v3.0* (GPL-3.0) License - see the [LICENSE.md](https://github.com/CoMPASS-Framework/CoMPASS-Labyrinth/blob/main/LICENSE.md) file for details.

## Contributors

- [Shreya Bangera](https://github.com/ShreyaBangera30)
- [Patrick Honma](https://github.com/pshonma)
- [Luiz Tauffer](https://github.com/luiztauffer)