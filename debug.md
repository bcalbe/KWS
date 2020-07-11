**# Problems**
- **Max_pooling cannot get input**
The second pooling layer cannot get the input since all kernels in the last conv layer are set to 0.
The key is why the APOZ in the last layer.
  - **Step 1: To input different classes to see the final output index.**
    - **Results**
  The results of input different classes shows that the index of the last layer are also set to zero. So that we can come to a conclusion that the actication of the last layer is very sparse. In all, the problem that need to be solved is that if how we choose the kernal if we meet such sparse activation.
    - 