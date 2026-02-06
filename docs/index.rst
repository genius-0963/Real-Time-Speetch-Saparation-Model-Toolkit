.. Real-Time-Speech-Separation-Model-Toolkit documentation master file, created by
   sphinx-quickstart on Thu Feb 6 17:47:00 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

==========
User guide
==========

Real-Time-Speech-Separation-Model-Toolkit is an open-source and all-in-one speech toolkit based on PyTorch.
This documentation provides install steps, tutorials and API documentation
necessary to help users develop their projects.

.. dropdown:: License considerations (Apache 2.0)

   Real-Time-Speech-Separation-Model-Toolkit is released under `Apache License, version 2.0 <https://github.com/genius-0963/Real-Time-Speetch-Saparation-Model-Toolkit/blob/main/LICENSE>`_. The Apache license is a popular BSD-like license.
   Real-Time-Speech-Separation-Model-Toolkit can be redistributed for free, even for commercial purposes, although you can not take off the license headers (and under some circumstances you may have to distribute a license document).
   Apache is not a viral license like GPL, which forces you to release your modifications to the source code. Also note that this project has no connection to Apache Foundation, other than that we use the same license terms.
   It is a community project, which means that discussions are engaged community-wide while decisions are taken by the project maintainers with respect to community views.
   There is no legal institution associated as an owner of Real-Time-Speech-Separation-Model-Toolkit. Furthermore, and due to the Apache License, anyone that would disagree with the way project is being run can fork it and start a new toolkit.

.. dropdown:: Referencing Real-Time-Speech-Separation-Model-Toolkit (BibTeX)

   If you use Real-Time-Speech-Separation-Model-Toolkit in your research or business, please cite it using the following BibTeX entry:

   .. code-block:: bibtex

      @misc{speechbrainV1,
         title={Open-Source Conversational AI with {SpeechBrain} 1.0},
         author={Mirco Ravanelli and Titouan Parcollet and Adel Moumen and Sylvain de Langen and Cem Subakan and Peter Plantinga and Yingzhi Wang and Pooneh Mousavi and Luca Della Libera and Artem Ploujnikov and Francesco Paissan and Davide Borra and Salah Zaiem and Zeyu Zhao and Shucong Zhang and Georgios Karakasidis and Sung-Lin Yeh and Pierre Champion and Aku Rouhe and Rudolf Braun and Florian Mai and Juan Zuluaga-Gomez and Seyed Mahed Mousavi and Andreas Nautsch and Xuechen Liu and Sangeet Sagar and Jarod Duret and Salima Mdhaffar and Gaelle Laperriere and Mickael Rouvier and Renato De Mori and Yannick Esteve},
         year={2024},
         eprint={2407.00463},
         archivePrefix={arXiv},
         primaryClass={cs.LG},
         url={https://arxiv.org/abs/2407.00463},
      }
      @misc{speechbrain,
         title={SpeechBrain: A General-Purpose Speech Toolkit},
         author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
         year={2021},
         eprint={2106.04624},
         archivePrefix={arXiv},
         primaryClass={eess.AS}
      }


We provide complete **Jupyter Notebook tutorials below** for beginners and
advanced users alike! You can view them in documentation, run them in Google
Colab or run them locally with
`Jupyter Notebook <https://jupyter.org/install>`_.

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   installation.md
   experiment.md
   guidance.md
   Contributing <https://github.com/speechbrain/speechbrain/blob/develop/docs/contributing.md>


.. toctree::
   :maxdepth: 1
   :caption: Tutorial notebooks

   tutorials/basics.rst
   tutorials/advanced.rst
   tutorials/preprocessing.rst
   tutorials/tasks.rst
   tutorials/nn.rst


.. toctree::
   :maxdepth: 1
   :caption: Tips & tricks

   audioloading.rst
   multigpu.md


API
---

.. toctree::
   :caption: API
   :hidden:
   :maxdepth: 3

   Core library (real_time_speech_separation_toolkit) <API/real_time_speech_separation_toolkit>
   HyperPyYAML (hyperpyyaml) <API/hyperpyyaml>

.. autosummary::

   real_time_speech_separation_toolkit
   real_time_speech_separation_toolkit.alignment
   real_time_speech_separation_toolkit.augment
   real_time_speech_separation_toolkit.dataio
   real_time_speech_separation_toolkit.decoders
   real_time_speech_separation_toolkit.inference
   real_time_speech_separation_toolkit.integrations
   real_time_speech_separation_toolkit.lm
   real_time_speech_separation_toolkit.lobes
   real_time_speech_separation_toolkit.nnet
   real_time_speech_separation_toolkit.processing
   real_time_speech_separation_toolkit.utils
   hyperpyyaml.core
