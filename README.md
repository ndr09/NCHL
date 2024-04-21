# NCHL

*Neuron-centric Hebbian Learning* </br>
*To appear at Gecco 2024* </br>
Andrea Ferigo, Elia Cunegatti, Giovanni Iacca <br>
University of Trento, Italy  <br>
[![arXiv](https://img.shields.io/badge/arXiv-2404.05621-b31b1b.svg)](https://arxiv.org/pdf/2403.12076.pdf)
 

```bibtex
@article{ferigo2024neuron,
  title={Neuron-centric Hebbian Learning},
  author={Ferigo, Andrea and Cunegatti, Elia and Iacca, Giovanni},
  journal={arXiv preprint arXiv:2403.12076},
  year={2024}
}
```
## Abstract

One of the most striking capabilities behind the learning mechanisms of the brain is the adaptation, through structural and functional plasticity, of its synapses. While synapses have the fundamental role of transmitting information across the brain, several studies show that it is the neuron activations that produce changes on synapses. Yet, most plasticity models devised for artificial Neural Networks (NNs), e.g., the ABCD rule, focus on synapses, rather than neurons, therefore optimizing synaptic-specific Hebbian parameters. This approach, however, increases the complexity of the optimization process since each synapse is associated to multiple Hebbian parameters. To overcome this limitation, we propose a novel plasticity model, called Neuron-centric Hebbian Learning (NcHL), where optimization focuses on neuron- rather than synaptic-specific Hebbian parameters. Compared to the ABCD rule, NcHL reduces the parameters from 5W to 5N, being W and N the number of weights and neurons, and usually N << W. We also devise a "weightless" NcHL model, which requires less memory by approximating the weights based on a record of neuron activations. Our experiments on two robotic locomotion tasks reveal that NcHL performs comparably to the ABCD rule, despite using up to ~97 times less parameters, thus allowing for scalable plasticity.


## Usage
In this repository, you will find the implementation of the Neuron Centric Hebbian Learning model. 
The model is written using the PyTorch library. To reproduce the results from the paper, you will also need the by bullet or Gymnasium libraries.
To set up the Python environment, we provide the ```requiremnents.txt``` file. 

## Paper Results and NCHL class usage
To replicate the results from the paper, you can use the ```task.py``` file. 
If you want to import the NCHL model into your codebase, you need to import the NHCL class from ```network.py```. 
The following code snapshot is the minimal code needed to import and initialize an instance of the NHCL class with one input, one hidden layer with two nodes, and 3 outputs. 
```
from network.py import NHCL
nchl= NHCL([1,2,3])
```

Then to set the Hebbian rules you have to execute the following code:
```
agent.set_hrules(hebbian_rules)
```
where hebbian_rules is a list of float with length nchl.nparams. 
To obtain the output of the nchl network you call the forward method. 
```
nchl.forward(input)
```
Note that the input variable must be a ```torch.Tensor```. 
Finally, to update the weights, you need to call the update function
```
nchl.update()
```

The abovementioned code procedure is replicated in the ```sample.ipynb```


## License 
This project is released under the MIT license.

## Contact
For any questions/doubts please feel free to contact us: andrea.ferigo@unitn.it, elia.cunegatti@unitn.it or giovanni.iacca@unitn.it
