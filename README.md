# archibrain
Synthesize brain architecture / models with artificial neural networks for cognitive tasks

Inspired by brain architecture, the machine learning community has recently developed various memory-augmented neural networks, that enable symbol and data manipulation tasks that are difficult with standard neural network approaches, see especially from Google Deepmind ([NTM](https://arxiv.org/abs/1410.5401), [DNC](http://www.nature.com/nature/journal/v538/n7626/abs/nature20101.html), [one-shot learner](https://arxiv.org/abs/1605.06065)). At the same time, models based closely on brain architecture, that perform experimentally-studied tasks, have existed for a while in the neuroscience community, notably from the labs of Eliasmith ([SPAWN](http://www.sciencemag.org/content/338/6111/1202)), O'Reilly ([PBWM](http://dx.doi.org/10.1162/089976606775093909)-[LEABRA-Emergent](http://www.colorado.edu/faculty/oreilly/research)), Alexander+Brown ([HER](http://dx.doi.org/10.1162/NECO_a_00779)), Roelfsema ([AuGMEnT](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004060)), Hawkins ([HTM](https://en.wikipedia.org/wiki/Hierarchical_temporal_memory)), and others. How these models compare to each other on standard tasks is unclear. Further, biological plausibility of these models is quite variable.

From the neuroscience perspective, we want to figure out how the brain performs cognitive tasks, by synthesizing current models and tasks, constrained by known architecture and learning rules,  From the machine learning perspective, we will explore whether brain-inspired architecture(s) can improve artificial intelligence (cf. copying bird flight didn't help build airplanes, but copying neurons helped machine learning).

We will utilize a modular architecture to:
1) Specify the model such that we can 'plug and play' different modules -- controller, differentiable memories (multiple can be used at the same time). We should be able to interface both the abstract 'neurons' (LSTM, GRU, McCullough-Pitts, ReLU, ...) but also more biological spiking neurons.
2) Specify Reinforcement Learning or other tasks -- 1-2AX, Raven progressive matrices, BABI tasks...

We will also explore different memory interfacing schemes like content or list-based as in the DNC, or Eliasmith's HHR, address-value augmentation, etc.

A larger goal will be to see if the synethesized 'network' can build models of the 'world' which generalize across tasks.

Mini-goal 1 (Aditya):
Understand and summarize various architectures. Explore current machine learning toolkits and figure out which one will be most suitable for our task.

Mini-goal 2 (Marco):
The Hierarchical Error Representation (HER) model by Alexander and Brown (2015, 2016), incorporating hierarchical predictive coding and gated working memory structures, is the most comparable to DeepMind's Differentiable Neural Computer (DNC).
(1) implement the DNC and the HER models using a standard machine learning toolkit (python front-end, with Theano/Keras/TensorFlow/mxnet); (2) benchmark the two architectures on similar tasks; (3) incorporate the best features of the two models (and any others) into a hopefully better (faster learning / better performing / more biologically plausible) model.

Mini-goal 3 (Vineet):
To be determined.
