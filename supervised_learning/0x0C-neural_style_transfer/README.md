# 0x0C. Neural Style Transfer

----
## Neural Style Transfer?
see [tensorflow.org](https://www.tensorflow.org/tutorials/generative/style_transfer)

Neural style transfer is an optimization technique used to take two images—a content image and a style reference image (such as an artwork by a famous painter)—and blend them together so the output image looks like the content image, but “painted” in the style of the style reference image.

This is implemented by optimizing the output image to match the content statistics of the content image and the style statistics of the style reference image. These statistics are extracted from the images using a convolutional network.

For example, let’s take an image of this dog and Wassily Kandinsky's Composition 7:

![Dog](https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg)

![Kandinski](https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg)


Now how would it look like if Kandinsky decided to paint the picture of this Dog exclusively with this style? Something like this?

![Neural Image](https://www.tensorflow.org/tutorials/generative/images/stylized-image.png)


----
## Documentation 


**Tutorial**

* [Neural Style Transfer: Creating Art with Deep Learning using tf.keras and eager execution](https://medium.com/tensorflow/neural-style-transfer-creating-art-with-deep-learning-using-tf-keras-and-eager-execution-7d541ac31398)

**deeplearning.ai**

* [What is neural style transfer?](https://www.youtube.com/watch?v=R39tWYYKNcI&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF&index=37)

* [What are deep CNs learning?](https://www.youtube.com/watch?v=ChoV5h7tw5A&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF&index=38)

* [Cost Function](https://www.youtube.com/watch?v=xY-DMAJpIP4&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF&index=39)

* [Content Cost Function](https://www.youtube.com/watch?v=b1I5X3UfEYI&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF&index=40)

* [Style Cost Function](https://www.youtube.com/watch?v=QgkLfjfGul8&index=41&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF)

**TensorFlow Documentation**

* [Eager execution](https://www.tensorflow.org/guide/eager)

**Scientific Paper**

* [A Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1508.06576.pdf)


----
## Changelog
* 28-Apr-2020 Creation

----
## Thanks
* [Coneja](https://github.com/macoyulloa)
