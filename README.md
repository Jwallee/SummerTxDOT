# **Narrative Image Analysis**
*Since this can't be containerized without getting admin approval, I'm going to list out everything that you need to run this code!*

*TxDOT*

*James Grant Robinett*

**Necessary Programs**
> 1. VsCode (or any other code editor)
> 2. Python (I'm using 3.11.4 64 Bit)
> 3. Pip (Should be included in your python install)

**Necessary Modules**
> 1. pdf2Image (Converts a pdf to a .jpg)
> 2. poppler (pdf text processor)
> 3. PIL (Pillow - adds processing power to python)
> 4. pytesseract (Module AND program - Does the image analysis)

**All of these need to be included in your path to run the code properly!**

---

## Instructions to run

With all modules and programs ready, it should be relatively simple to run. Located in this folder is a python script called `narrativeAnalysis`. To process your report pdfs, place them in the folder labeled `crashes`. Do not worry what you name them, all images and text files created will be named the same as the pdf automatically. With these in place, just click on the python script and run!

The outputs should appear in the folders `cropped`, `liftedText` and `whole`. `cropped` containes the cropped images only containing the information present in the narrative of the pdf. `liftedText` contains the processed text files for each of the narrative images analyzed. (I am not sure how accurate it is, but from my tests it is very very accurate). Lastly, `whole` contains the images of the entire second page of the CR-3 report. You can use this to compare or check if anything is out of order.

---

## Why did I do this?

1 - No narrative is processed in our data collection. This makes it harder to collect what actually happened at the scene of a crash.

2 - All of the narratives are saved in a non-text collectable pdf image form, making it so you cannot easily scan the information off the narrative.

3 - Image processing is needed. Luckily, pytesseract is a great program that does fairly well with image text analysis.

4 - There is a lot of noise with the entire pdf being analyzed. Cropping and reducing analysis to just the narrative box will help analysis.

5 - Refine the image text analysis. With this refined and tweaked, we can then move on to sorting these common phrases and matching them with report data collected.

6 - With these links made, we can now feed this large-scale data into a machine learning algorithm. First small, then testing with a large number of reports.

7 - Once this is finished, we should be able to throw it just the narrative and see where that gets us with our analysis. Maybe could properly check some of the entered information and check and see if it was properly filled out.

**Problems**
- Incorrect text analysis (FIXED)
- Hard distinction of injury status when the officer does not use many words (ex - noninjured vs minor injury, can be not written in narrative.)
- Need to see what specifically to analyze.

**THINGS THAT NEED TO BE AUTOMATED**

1. Downloading of report pdfs
2. Saving cropped images as their own hash codes (FINISHED)
3. Load these images and informational bits into a folder (possibly database in the future. Kubernetes?) (FINISHED)


*To be fair, this is a lot easier said than done. I will try my best to see if this process is useful or not but it may come to a dead end due to my inexperience or lack of usability*.
