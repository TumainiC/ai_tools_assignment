# Part 1: Theoretical Questions

## Question 1: Explain the primary differences between TensorFlow and PyTorch. When would you choose one over the other?

To put it simply, Pytorch is easier to use than Tensorflow. 

Tensorflow works by using **static computation graphs**. The entire computation grapgh is defined before execution, then optimized and compiled (You plan everythong first then build it all at once). 

Conversely, Pytorch works using **dynamic computation graphs**. The graph is built on the fly during runtime (You can mold and change things as you go).

The differences:

1. Tensorflow uses graph mode and requires superior debugging tools like tf.debugging or TensorBoard. Pytorch on the other hand uses standard python debuggers since operations behave like native python. 

2. Tensorflow is better at deploying at scale and in production or when mobile is required, whereas pytorch is better when experimenting and prototyping or when debugging complexity is high. 


## Question 2: Describe two use cases for Jupyter Notebooks in AI development.

**Answer:**

#### Use Case 1:
**Conducting Exploratory Data Analysis and Model prototyping**

In jupyter notebooks you can write code and see results immediately as well as write notes within your code using markdown. You can explore your data to understand it and make sense of patterns present in the data. You can also make changes to your data and experiment with different approaches. 

They are preferred because you can execute and modify cells in segments without having to rerun the entire script. You also get immediate visual fedback from run cells, in a matter of seconds. 

#### Use Case 2: 
**Educational Demos and Reproducability**

Jupyter notebooks serve as executable documentation in the sense that they combine code, results and explanations into one single shareable document. Code cells can be modified and rerun to see different results and the exeperiment can be rerun.


## Question 3: How does spaCy enhance NLP tasks compared to basic Python string operations?
**Answer:**
spaCy provides deep linguisting understanding that basic string operations do not have. Its understand words as verbs, tenses of words and not just the simple structure of how many elements are in a word or how to koin two words like string operations. spaCy is essential for tasks requiring linguistic understanding: information extraction, question answering, document classification, and machine translation preprocessing.