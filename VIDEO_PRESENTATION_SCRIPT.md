# 5–8 Minute Video Presentation Script (4 Team Members)
**Project:** CNN-Based Helmet and Mask Detection  
**Tone:** Natural, conversational, simple student English (easy to speak out loud, sounds authentic and spontaneous, not robotic or memorized).

---

## ⏱️ Video Overview & Speaker Breakdown (Total: ~7 Minutes)

| Presenter | Section | Slides / Action | Approx. Time |
| :--- | :--- | :--- | :--- |
| **Member 1** | Introduction, Real-World Problem & Dataset | **Slides 1, 2, 3** | ~1.5 – 2 mins |
| **Member 2** | Mathematical Concepts & CNN Flow | **Slides 4, 5, 6, 7** | ~2 mins |
| **Member 3** | Live Code & Terminal Walkthrough in IDE | **Live VS Code Screen Share** | ~2 mins |
| **Member 4** | Visual Feature Progression, Results & Wrap-up | **Slides 8, 9, 10** | ~1.5 – 2 mins |

---

## 🎬 SCRIPT START

---

### 🎙️ MEMBER 1 (Slides 1, 2, 3) — Intro, Problem & Dataset

#### [Slide 1: Title Slide — "HELMET / MASK DETECTION USING A CNN"]
> *"Hello everyone, and welcome to our presentation. Today, our team is presenting our Deep Learning project on **Helmet and Mask Detection using a Convolutional Neural Network**.*  
> *Our team consists of four members: myself, Member 2, Member 3, and Member 4. In this video, we're going to break down the real-world problem, the mathematical concepts behind CNNs, do a live code walkthrough right in our IDE, and show you our visual results."*

*(Click to Slide 2)*

#### [Slide 2: "IS SAFETY GEAR BEING WORN?"]
> *"Let's start with why this actually matters. In hazardous areas like construction sites, industrial plants, or even on busy roads, checking whether people are wearing helmets or masks is crucial for safety.*  
> *Right now, most safety monitoring is done manually by security guards or supervisors. But human checking gets tiring, it's slow, and you can't have someone watching dozens of camera feeds all day.*  
> *So our objective was to build an automated, lightweight binary classifier. It takes in camera images and immediately answers one simple question: **Is the person wearing their safety gear, or not?**"*

*(Click to Slide 3)*

#### [Slide 3: "326 PHOTOS, TWO BALANCED CLASSES"]
> *"To train our model, we curated a dataset of 326 real-world photos. Here on Slide 3, you can see some sample images from our test set.*  
> *The top row shows people wearing helmets while riding or working. The bottom row shows people without any headgear.*  
> *We made sure the dataset is strictly balanced—50% with helmet/mask, and 50% without. We split it into 70% for training (that's 226 photos), 15% for validation, and a separate 15% (50 photos) kept completely aside for testing.*  
> *Now, how does a neural network actually look at these photos and learn features? I'll pass it over to Member 2 to explain the core concepts."*

---

### 🎙️ MEMBER 2 (Slides 4, 5, 6, 7) — Mathematical Concepts & CNN Flow

#### [Slide 4: "AN IMAGE IS A GRID OF NUMBERS"]
> *"Thanks, Member 1! Hi everyone. Before we jump into layers, let's understand how a computer sees an image.*  
> *To us, it's a person wearing a helmet. But to the CNN, an image is just a 3D grid of numbers—specifically, an RGB tensor of size 128 by 128 pixels with 3 color channels: Red, Green, and Blue.*  
> *Each pixel value ranges from 0 to 255. To detect shapes and edges, we use something called a **kernel**—which is just a small 3 by 3 matrix of numbers, like the Sobel edge filter shown on the right. Think of the kernel as a small template that searches for specific visual cues."*

*(Click to Slide 5)*

#### [Slide 5: "SLIDING THE KERNEL MAKES A FEATURE MAP"]
> *"Now on Slide 5, you can see the **Convolution** operation.*  
> *The 3 by 3 kernel slides over the image step by step. At every position, it multiplies overlapping numbers element-by-element and adds them up—that's the formula you see on the right.*  
> *If an edge or pattern in the image matches the kernel, the output number is high. As the kernel scans the whole image, it produces a brand-new 2D grid called a **Feature Map**, which highlights where edges and lines are located."*

*(Click to Slide 6)*

#### [Slide 6: "RELU KEEPS POSITIVES, POOLING KEEPS THE STRONGEST"]
> *"After convolution, we apply two important operations shown here on Slide 6.*  
> *First is **ReLU**—Rectified Linear Unit. Its formula is super simple: $f(x) = \max(0, x)$. Basically, if a number is negative, it gets turned into zero, which represents black. If it's positive, it stays as it is. This gets rid of unnecessary background noise and gives our network non-linear learning capability.*  
> *Second is **Max Pooling**. Here we take a 2 by 2 window and slide it across the feature map. In each 2 by 2 box, we only keep the largest number—for example, between 1, 7, 5, and 3, we pick 7. This cuts the image height and width in half—from 128 down to 64—which speeds up computation and helps the network recognize objects even if they shift slightly."*

*(Click to Slide 7)*

#### [Slide 7: "FROM PIXELS TO ONE SIGMOID DECISION"]
> *"Finally, on Slide 7, this is how our entire pipeline comes together.*  
> *The image passes through multiple Conv, ReLU, and Pooling layers. The deep features are flattened into a 1D vector and fed into Dense layers.*  
> *At the very end, we use a **Sigmoid activation function**. The Sigmoid outputs a single probability score between 0 and 1. We set our decision threshold at 0.5: if the score is less than 0.5, it classifies the person as **Wearing helmet/mask** (Class 0). If it's 0.5 or higher, it classifies as **Not wearing** (Class 1).*  
> *Now, Member 3 is going to take over the screen and show you our actual Python implementation running live in the IDE."*

---

### 🎙️ MEMBER 3 (Live Screen Share in VS Code / IDE) — Code Walkthrough & Demo

#### [Action: Switch screen share from PPT to VS Code / IDE editor]

> *"Thanks, Member 2. Hello everyone, I'm Member 3, and I'll walk you through our Python code and run a live demonstration right here in our IDE.*  
> *(Points to file explorer)*  
> *Our project is structured into three clean, explainable scripts: `train.py`, `visualize.py`, and `predict.py`.*

#### [Open `train.py` in editor, scroll to the model definition lines ~40–55]
> *"First, here in `train.py`, we built our CNN using TensorFlow and Keras Sequential API.*  
> *We start with an Input shape of 128 by 128 by 3, followed by a Rescaling layer to normalize pixel values from 0 to 255 down to 0 to 1.*  
> *Then we have three convolutional blocks:*  
> *First, Conv2D with 32 filters, ReLU activation, followed by a 2 by 2 MaxPooling.*  
> *Next, a second Conv2D with 64 filters and MaxPooling.*  
> *And a third Conv2D with 128 filters and MaxPooling.*  
> *After extracting features, we Flatten the maps, pass them through a Dense layer of 128 neurons with 30% Dropout to avoid overfitting, and finish with a single Dense neuron using Sigmoid activation."*

#### [Open the IDE terminal inside VS Code]
> *"Now let's see the model in action. First, let's run our prediction script:*  
> `python predict.py dataset/test/with_helmet_mask/with_helmet_mask_11.jpg`*  
> *(Press Enter and let the output print)*  
> *You can see it instantly loaded our trained model, processed the test image, and predicted: **Wearing helmet/mask** with an 81.9% confidence score.*  
> *If we run it on a no-helmet image:*  
> `python predict.py dataset/test/without_helmet_mask/without_helmet_mask_12.jpg`*  
> *(Press Enter)*  
> *It correctly predicts: **Not wearing helmet/mask** with 95.1% confidence."*

#### [Run visualize.py in terminal]
> *"Next, to satisfy the ABA-2 assignment requirement of demonstrating the visual progression from raw image to classification, we wrote `visualize.py`.*  
> *Let's run it:*  
> `python visualize.py`*  
> *(Press Enter and show terminal output)*  
> *Notice what the terminal prints out: it extracts the exact 3x3 learned filter weights, computes the linear convolution dot products, applies the ReLU threshold, downsamples through max pooling, and generates our visual progression diagrams.*  
> *Now, Member 4 will take us back to the slides to examine these visual feature maps in detail and review our experimental results."*

---

### 🎙️ MEMBER 4 (Slides 8, 9, 10) — Visual Features, Results & Conclusion

#### [Action: Switch screen share back to PPT on Slide 8]

#### [Slide 8: "HOW THE CNN SEES: EDGES TO SHAPES TO OBJECTS"]
> *"Thanks, Member 3! Hi everyone, I'm Member 4.*  
> *Here on Slide 8, you can see the exact output that Member 3's visualization script generated for a test image of a scooter rider.*  
> *On the left card, look at the 7-stage progression:*  
> *1. The original 128x128 image.*  
> *2. The learned 3x3 kernel weights shown as a color heatmap.*  
> *3. The linear convolution response.*  
> *4. The resulting feature map.*  
> *5. The ReLU step—notice how the background negative numbers turned completely black, leaving only the sharp edges of the rider and vehicle glowing.*  
> *6. The 2x2 pooling step, which preserves those strong edges at half the resolution.*  
> *And 7. The final classification verdict.*  
> *On the right card, you see how features evolve across layers:*  
> *Conv1 captures fine edges and textures.*  
> *Conv2 combines those edges into curves and silhouettes.*  
> *And Conv3 highlights abstract regions—specifically focusing on the head and helmet area."*

*(Click to Slide 9)*

#### [Slide 9: "78% ACCURACY ON UNSEEN PHOTOS"]
> *"Moving to Slide 9, here are our experimental results.*  
> *We evaluated the trained model on 50 completely unseen test images—25 with helmet and 25 without.*  
> *Overall, our model achieved **78.00% test accuracy** with a test loss of 0.5148.*  
> *Looking at our confusion matrix in the bottom card:*  
> *Out of 25 helmet images, 19 were correctly identified as True Positives.*  
> *Out of 25 no-helmet images, 20 were correctly identified as True Negatives.*  
> *The top charts show our training and validation curves. You can see the loss steadily decreasing and validation accuracy climbing steadily, proving that our network learned meaningful generalizable patterns rather than just memorizing."*

*(Click to Slide 10)*

#### [Slide 10: "WHAT WE BUILT, AND WHAT'S NEXT"]
> *"Finally, to wrap up our project on Slide 10:*  
> *What we achieved: We designed, trained, and validated an end-to-end binary CNN classifier, successfully visualized the internal feature extraction mechanics, and proved that a compact CNN can reliably detect safety gear.*  
> *For future work: We plan to expand the dataset with more lighting variations, experiment with transfer learning models like MobileNet for even faster mobile inference, and integrate this into a live RTSP webcam stream for automated workplace alerts.*  
> *That concludes our presentation. Thank you so much for your time, and we'd be happy to answer any questions!"*

---

## 💡 Quick Tips for the Recording

1. **Keep it natural:** If you stumble on a word, don't worry or start over—just correct yourself and keep going, exactly like in a live classroom viva.
2. **Member 3 (IDE Screen):** Make sure your terminal font size is zoomed in slightly (`Ctrl + +` in VS Code) so your code and command outputs are crisp and readable on video.
3. **Pointers:** Use your mouse cursor to point at the diagram cells on Slides 5, 6, 8, and 9 while speaking about them.
