# Image classify using own images with tensorflow

####Introduction:
  If you are the new user in tensorflow (just like me two weeks ago), you can see what is the fantasty feature in [here] (https://www.tensorflow.org/versions/r0.10/tutorials/image_recognition/index.html)
  
  I have already test it:
  ![alt tag](https://github.com/haobangpig/Image-classify-using-own-images-with-tensorflow-/blob/master/images/lion_classify.png)
  
  It is quit easy, just installed the tensorflow and see the [website] (https://www.tensorflow.org/versions/r0.10/tutorials/image_recognition/index.html) . Running at your terminal.
  But how to input your own images and  and output the possibility after training it.
  
  So in this project, it will make a really simple demonstrate about it.
  
###Main Idea:
  This is about using your own images(JPEG or PNG) and created labels with tensorflow.
  
###Steps for studying:
  
|Step|To do|
|------|----|
|1|Understand the key concepts|
|2|Understand the basic tutorial in Tensorflow(MNIST)|
|3|Modify the basic tutorial (MNIST) to output the prediction and each of the labels possibility |
|4|Reformat the images(JPEG or PNG) to the specification format|
|5|Using step three's function to output the result|



####Step One:
#####*Using google or baidu or anything you can, try to search everything about the four key words that we will use in this project.*
  Key Concept:
  * Deep Learning & Neural Network
  * Softmax Regression 
  * Gradient descent
  * CNN(Convolutional Neural Network)

####Review:
#####*Pleae use your own language to illustrate:*
  
  (If you can make the others who never hear these concepts to understand with a simple example, that mean you had already understand these.)
  * What is the Deep learning & Neural Network
  * How to create the Softmax Regression Model 
  * What does the Gradient descent usage doing
  * Please make a simple demonstration to explain what is the CNN and what is the biggest feature of it. 
  
  
  
####Step Two:
#####*[Go to The tensorflow tutorial website and download the original the code and try to understand each of line](https://www.tensorflow.org/versions/r0.10/tutorials/mnist/beginners/index.html).*

  You may follow some steps in here to understand more about the tensorflow
  * [HelloWorld in tensorflow] (https://www.tensorflow.org/versions/r0.10/get_started/index.html) 
  * [Placeholder] (http://learningtensorflow.com/lesson4/)  
  * [Feed] (https://www.tensorflow.org/versions/r0.10/how_tos/reading_data/index.html#feeding)
  * [Creation, Initialization, Saving, and Loading] (https://www.tensorflow.org/versions/r0.10/how_tos/variables/index.html)
  * [Tf.argmax] (https://www.tensorflow.org/versions/r0.10/api_docs/python/math_ops.html#argmax)
  * [Output the result and possibilty] (https://github.com/tensorflow/tensorflow/issues/97)



####Step Three: Understand this example and edit the output as you wish
#####*In this step, it is really important in this project and it will help you to understand what is the structure in the tensorflow.*

##### At the beginning, I wanna use its own test data and output the possibility after I insert a test image. But I was totally no idea how to do it. But luckly, I found this blog: [Using TensorFlow to create your own handwriting recognition engine] (https://niektemme.com/2016/02/21/tensorflow-handwriting/). Thanks the writer(Niek Temme) who help me solved this problem. I just directly upload his code on this project. Above that, I add some functions in his source code to content with my demand.  


The logical in the source core is quite sample, I will illustrate with the pseudocode and you may check the source code in
[this file] (https://github.com/haobangpig/Image-classify-using-own-images-with-tensorflow-/tree/master/tensorflow-mnist-predict).




#####Create_model

```
1>import the dataset 
2>create the softmax regression model (placeholder)
3>Set the GradientDescent function
4>session initialize 
5>traing
6>save as .ckpt file formate (Saver())
7>output the training_test posibility
```

#####Prediction

```
1>import the test image from your computer
2>use imageprepare() to reformat the input image
3>use predictint() to give the result and the possibility.
```
Prediction.predictini() :
```
1>Define the softmax regression model
2>session initial
3>load the  .ckpt file 
4>use _evel function to get the prediction
5>use sess.run function to get each label's possibility
```

And the result will be like this :
  ![alt tag](https://github.com/haobangpig/Image-classify-using-own-images-with-tensorflow-/blob/master/images/handwriting_recognize.png)
  


####Step Four:
#####*In this step, you need to know the CNN and how it works.*
Check this [course](https://www.udacity.com/course/deep-learning--ud730)to understand more about the CNN.
And check the assignment('Assignment one' and 'Assignment four') in this course.
'The assignment one' are the great example to illustrate how to convert the images to the dataset and store into the .pickle file.
'The assignment four' give the example that how to set the dataset to the images shape. See the source code and you may understand.
'The assignment one' use these labels ={'A','B','C','D','E','F','G','H','I','J'}. In each of the label, there are bunchs of jpg or png format on it. So you can change the label name to your label and put a plenty of the images on it.



####Step Five:
#####*In this step, We just need to use the step three souce code and add them in 'The assignment Four'. And we will got what we want.*

See the source code and you may understand it.

NOTICE:
-If you input a image that you wanna test, it may have some problems. If that happened, please use the imageprepare() function in step three to reformat the input image. 

- 'AS4.py'  will be trained the model each time when you running it and then test the input file(filename). So if you wanna test the simple directly, try to use the Saver() function to save as '.ckpt' and retore it when you test the image(Like step three did). 

![alt tag](https://github.com/haobangpig/Image-classify-using-own-images-with-tensorflow-/blob/master/images/The%20last%20result.png)
Good luck for everything.

If you wanna contact me, try this way : haobangpig@gmail.com
