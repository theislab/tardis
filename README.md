# TarDis: Targeted Disentanglement

Addressing challenges in domain invariance within single-cell genomics necessitates innovative strategies to manage the heterogeneity of multi-source datasets while maintaining the integrity of biological signals. We introduce TarDis, a novel deep generative model designed to disentangle intricate covariate structures across diverse biological datasets, distinguishing technical artifacts from true biological variations. By employing tailored covariate-specific loss components and a self-supervised approach, TarDis effectively generates multiple latent space representations that capture each continuous and categorical target covariate separately, along with unexplained variation. Our extensive evaluations demonstrate that TarDis outperforms existing methods in data integration, covariate disentanglement, and robust out-of-distribution predictions. The model’s capacity to produce interpretable and structured latent spaces, including ordered latent representations for continuous covariates, enhances its utility in hypothesis-driven research. Consequently, TarDis offers a promising analytical platform for advancing scientific discovery, providing insights into cellular dynamics, and enabling targeted therapeutic interventions.

## Paper

The methodologies, experiments, and results discussed in this repository are detailed in our paper on bioRxiv. You can read and cite the paper using the following link:

[Read the TarDis paper on bioRxiv](https://www.biorxiv.org/content/10.1101/2024.06.20.599903v1)

## Notes

### Codebase Updates

The TarDis codebase is under continuous development to incorporate the latest research findings and community feedback. We aim to update and tidy up the repository regularly to improve functionality and user experience. Users can track changes and updates through the repository's commit history or the release notes.

### Contact and Support

For assistance with the TarDis model, suggestions for enhancements, or any other inquiries, please do not hesitate to contact the lead developer, Kemal Inecik. You can reach him via email at [k.inecik@gmail.com](mailto:k.inecik@gmail.com). If you encounter any issues or if you have features you would like to suggest or contribute, please file an issue or pull request on GitHub.

### Acknowledgements

We appreciate the efforts of all contributors who have helped in refining the TarDis model and extending its capabilities. Special thanks to the community for providing valuable feedback and to our funding agencies for supporting this research.
