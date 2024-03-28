This repo aims to classify skin cancer images into its various types. I have used two models to achieve this..
  - MobileNet : Achieved recall score about 70 %. Model is good to use but took about 1:34 hours for 10 epochs and on training further accuracy is decreasing which  means training data is over-fitting..
  - EfficientNet : Achieved recall score about 75%. Two epoch took about 20 minutes. On further training, accuracy started decreasing.
    - I have used b3 model of efficientNet as it matchs best to the given data size and complexity..
   
- Further for loss function, I have used CrossEntropyLoss fucntion to calculate. We can use other criterions also to improve recall score..
  
