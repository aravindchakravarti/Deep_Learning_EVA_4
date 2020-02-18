In this project, the Goal is to achieve 99.4% accuracy with less than 10k Parameter. Lets build the network sequentially and see if can achieve our target.

Lets try these things first.

# Try 1:
## Target
* Make code more 'Readable' by adding some visualizers. Such as, Training and Test accuracy plots.
* Use drop out and check the accuracy.

## Result
* I didn't hit the required accuracy
* On an average test accuracy was around 98.7%
* Around 13k parameters

## Analysis
* By looking at the plot we can say that, there no over-fitting. But, training and test accuarcy plot is not very smooth. That means learning rate is too high at the end.
* Probably, after GAP, I may need to add 1x1x10 and then what happens. Because, after GAP, we are allowed to add FC, as we know, by the time of GAP, more or less, we have class label data.

## Code is available at:
https://drive.google.com/file/d/1yDT9P4jc6ITXiTGnnex7boq_JblC_z_I/view?usp=sharing


------------------------------------------------------------------------------------------------------------

# Try 2:
## Target
* After GAP, add a FC
* Hit <10k parameter
* Do have the accuracy at-least 98%

## Result
* Successfully added FC after GAP
* 9772 parameters
* Accuracy = 98.95 in last 5 epochs (rough average)

## Analysis
* Decreasing learning helped to smoothen the curves (<- It is a good sign)
* Even with <10k parametes, hitting 98.95% accuracy is good. But not enough to hit 99.4%
* Next step is to use master tool -> Augmentation!!

## Code is available at:
https://colab.research.google.com/drive/1WJy6A6LNBH2-BRipTBInKbOzsRhuxlfx



------------------------------------------------------------------------------------------------------------

# Try 3:
## Target
* Use augmentations and see where I can get
* Change the model little bit/make it more efficient 

## Result
* Parameters = 8.6k 
* Accuracy = 98.1% in last 5 epoch (approx)
* Augmentation = Used

## Analysis
* MNIST is a very easy dataset. We do not need many kernels in first few layers. As, barely there are horizontal and vertical edges. Hence, decided to use just 6 kernels in beginning. 
* Proabably I should use LR Schedular to hit 99.4% atleast 2-3 times in last 5 epochs

## Code is available at:
https://colab.research.google.com/drive/1Mv1G_W0GsxoiPGn61q_NyqMqQ3HNlh6R


------------------------------------------------------------------------------------------------------------


# Try 4:
## Target
* Must hit 99.4% accuracy atleast 3 times in 15 epochs
* Should I really use 1x1 kernel? May not be, because we have like 10-14 channels in first block. May not be required.
* Use learning rate schedular

## Result
* Achieved 99.4 in 1 epoch! (12th epoch)
* Parameters = 9.3k parameters

## Analysis
* Readuced the learning rate to 0.005 with StepLR of gamma 0.5/10 epochs
* Reduced (almost removed the dropout) - Why? We are using the augmentations. So, dropout may not do much help.
* Removed 1x1 kernel at this moment. Why?- Because have about 10-14 channels in first block. 1x1 may not help too much. 


## Code is available at:
https://colab.research.google.com/drive/1VrXTDA58W8JtBhCtvCv_S08YwT-Jj6Rl


------------------------------------------------------------------------------------------------------------

# Try 5:
## Target
* Must hit 99.4% accuracy atleast 3 times in 15 epochs
* Change augmentation stratergy if not able to acheive the required accuracy

## Result
* Achieved 99.4, three times
* Used shear augmentation stratergy as well
* Number of parameters 9302 parameters

## Analysis
By looking at the dataset, I felt like shear transform also should work along with rotation. Because the straight lines drawn by human hands will not be 90 degrees, but will have some inclination
When we see the train accuracy it is about 99.0% and test accuracy is around 99.4%. This shows that network has very good capacity. If we train a little extra, probably we may achieve much more accuracy
Code is available at:

## Code is available at:
https://colab.research.google.com/drive/1mUwuzT9dvQRV89GskhuHYy1VXttPypYD


------------------------------------------------------------------------------------------------------------
