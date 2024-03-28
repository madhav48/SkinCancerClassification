This repo aims to classify skin cancer images into its various types. I have used two models to achieve this..
  - MobileNet : Achieved recall score about 70 %. Model is good to use but took about 1:34 hours for 10 epochs and on training further accuracy is decreasing which  means training data is over-fitting..
  - EfficientNet : Achieved recall score about 75%. Two epochs took about 20 minutes. On further training, accuracy started decreasing.
    - I have used b3 model of efficientNet as it matches best to the given data size and complexity..
   
- Further for the loss function, I have used the CrossEntropyLoss function to calculate. We can use other criterions also to improve recall score..
