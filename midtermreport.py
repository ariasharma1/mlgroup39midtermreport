import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the title of your app
st.title("Cardiovascular Disease Classification Project")
st.subheader("Introduction/Background")
st.write("Our project focuses on classifying cardiovascular disease using medical data, including biometrics like gender, height, weight, and blood pressure. A study that used machine learning to predict whether a patient had heart disease tested several machine learning algorithms, and “the results showed that RF (91.80%) had the highest accuracy in predicting heart disease, followed by NB (88.52%) and SVM (88.52%)” [1]. Given these results, we will prioritize RF and SVM in our supervised learning approach. Our dataset includes 70,000 entries, each with 12 features—11 biometrics and 1 indicating cardiovascular disease.")

st.markdown("[Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)")

st.subheader("Problem Definition")
st.write("According to the WHO, “cardiovascular diseases (CVDs) are the leading cause of death globally, taking an estimated 17.9 million lives each year” [2]. By creating a model to classify whether a person has cardiovascular disease, preventative actions can be taken to mitigate the devastating effects of cardiovascular diseases.")

st.subheader("Methods")
st.write("Upon initial inspection, our dataset is primarily composed of numerical values, so we decided to use Pandas and Numpy library. ")
st.write("Our 3 data preprocessing methods are as follows:")
st.markdown("""
1. **_Data Cleaning_**
    - Duplicate Values: No duplicates in our dataset
    - Missing Values: No missing values in our dataset
""")

st.markdown("""
    - **Height:** 
            -Since our data focuses on individuals in their middle age, spanning a height range from 5’3" (160 cm) to 6’6" (198 cm) for men and the 4’10’’ (147cm) to 6’1’’ (185cm) for women, according to the 2007-2008 census data [3]. It's worth noting that the cumulative percentage of the population falling outside this range is roughly 1%. Consequently, we classify these data points as outliers and exclude them from our dataset. The box plot of data after removal shows lower variability with a few tall individuals.
""")
            
st.image("/Users/ariasharma/Desktop/height1.png", use_column_width=True)
st.image("/Users/ariasharma/Desktop/height2.png", use_column_width=True)

st.markdown("""
    - **Weight:** 
            -Since our data focuses on individuals in their middle age, spanning a height range from 5’3" (160 cm) to 6’6" (198 cm) for men and the 4’10’’ (147cm) to 6’1’’ (185cm) for women, according to the 2007-2008 census data [3]. It's worth noting that the cumulative percentage of the population falling outside this range is roughly 1%. Consequently, we classify these data points as outliers and exclude them from our dataset. The box plot of data after removal shows lower variability with a few tall individuals. 
""")
st.image("/Users/ariasharma/Desktop/weight1.png", use_column_width=True)
st.image("/Users/ariasharma/Desktop/weight2.png", use_column_width=True)


st.markdown("""

    - **Blood Pressure:** 
            -The systolic and diastolic blood pressures range from normal 90/60 mm Hg to stage 2 hypertension 160/100 mm Hg [5]. Any pressure outside of this range is considered life threatening emergency. So, we excluded those datapoints with buffer of +/- 30mm Hg.  """)
st.image("/Users/ariasharma/Desktop/bp1.png", use_column_width=True)
st.image("/Users/ariasharma/Desktop/bp2.png", use_column_width=True)


st.write("We considered applying Principal Component Analysis to the original and the validated datasets to visualize any possible outlier groups. Based on plot PCA plot using 1st and 2nd components, there is no big cluster of outliers so all remaining datapoints are kept.")
st.image("/Users/ariasharma/Desktop/graph1.png", use_column_width=True)

st.markdown("""
2. **_Feature Engineering_**
""")

st.markdown("""
    -In the original dataset, ages were recorded in days, leading to unnecessary variation. To address this, we converted the unit from days to years. The boxplot reveals that our data primarily focuses on individuals in their middle age. 
""")
st.image("/Users/ariasharma/Desktop/featureengineering.png", use_column_width=True)
st.markdown("""
    -We combined height and weight to calculate BMI to further evaluate the validity of our dataset. Most BMI reports don’t record BMI that is higher than 60 so we excluded those data from as well. """)

st.markdown("""
3. **_Standardization of Values inluding BP, Height, Weight_**
""")
st.markdown("""
    -Certain features exhibit a broad range of values, such as blood pressures, which could dominate the classification algorithm. To balance their impacts on distance calculations, we plan to scale these features with mean of 0 and stdev of 1, utilizing the scikit-learn library """)

st.write("After data-processing, the correlation matrix is as below. Since there are only 12 features, we decided to keep them all. ")

st.image("/Users/ariasharma/Desktop/correlationmatrix.png", use_column_width=True)

st.write("Upon examining the correlation matrix between features, it shows that the majority do not exhibit high correlation, suggesting our dataset is complex and potentially contains both linear and non-linear relationships. Given the dataset's size of 70,000 entries and its complexity, the Random Forest algorithm initially seemed like the ideal choice for our project. However, we still want to explore other, simpler algorithms to verify whether our initial assumption about the dataset's large size and complexity is accurate. Two algorithms under consideration are logistic regression, which can serve as a basic benchmark but assumes the linear relationship between features, and SVM, which is effective for high-dimensional spaces but slow on large datasets. We chose to implement linear regression first to achieve a general baseline for performance that we could compare other models to. We then proceeded to have working implementations of our other two proposed supervised learning models: SVM and Random Forest. We found that SVM currently has a much poorer performance than both linear regression and random forest, and we plan to improve the performance of the SVM algorithm by tuning parameters and experimenting with different kernels. We found that linear regression and random forest return very similar results, which will be discussed further in the next section. We chose our supervised learning models carefully, specifically choosing linear regression to obtain a good baseline for performance, choosing SVM because we have a high-dimensional feature space, and choosing random forest because of its flexibility and ability to model nonlinear relationships. We have also brainstormed which unsupervised models we will use. We have already used PCA to reduce the dimensionality of the feature space, but we have used this solely for visualization up to this point, rather than using it to simplify our data. Therefore, we are planning to employ PCA in the future more directly. We are also considering using k-means clustering, specifically with two clusters, since this is a binary classification problem.  ")

st.subheader("(Potential) Results and Discussion")
st.write("For this project, we will evaluate our ML model using several quantitative metrics.")
st.markdown("""
    - **Accuracy**
            -To measure the proportion of true results (true positives & true negatives) among the total cases. Our goal is to achieve accuracy exceeding 85%, which would indicate strong model performance. Furthermore, we are looking to measure false positive and false negative counts, since those values are crucial when providing a medical diagnosis. 
    - **F1 Score**
            -To balance precision and recall. Using the F1 score would help us minimize false positives while accurately identifying patients with cardiovascular disease. We aim for an F1 Score of at least 0.75. 
    - **Receiver Operating Characteristic**       
            -Area Under Curve (ROC-AUC): To assess the model’s ability to accurately distinguish between patients with and without cardiovascular disease. We aim for our mode to have a  ROC-AUC score above 0.80.         
""")

st.write("For the supervised algorithms we have implemented thus far, we have unfortunately not been able to reach the goals we listed above. Currently, our SVM implementation performs far worse than the other two models we have implemented. It has returned an accuracy score of 0.53, an F1 score of 0.64, and an ROC-AUC score of 0.59. We believe that these results are due to the implementation of our SVM algorithm that has not quite been optimized yet. Specifically, we will be trying to optimize parameters and experiment with different kernels in the future to find a better solution. We will be trying to experiment with different values of the regularization constant and kernel coefficient. Our linear regression and random forest models, however, returned encouraging results. Our linear regression model returned an accuracy score of 0.72, an F1 score of 0.71, and an ROC-AUC score of 0.78. Our random forest model returned very similar results, specifically an accuracy score of 0.73, an F1 score of 0.71, and an ROC-AUC score of 0.79. The results of both of these models fall just short of our initial goals, and we will be trying to improve them in the future by cleaning the dataset further to remove outliers and noise. Our most important goal in the future is the implementation of unsupervised learning methods. As mentioned earlier, we have already implemented PCA, but we have only done so for visualization purposes, and we have not yet taken full advantage of its data simplification capabilities. We can surely obtain good results from PCA by removing noise and overfitting concerns. Furthermore, we would like to also employ one more unsupervised learning technique, if possible. For now, we have mentioned k-means clustering as a candidate, but we will brainstorm more ideas in the future.  ")

st.write("Our main goal is to develop a reliable model that accurately predicts cardiovascular disease, to improve early detection and intervention efforts. ")

st.markdown("""
## References

1. **A. A. Ahmad and H. Polat**, “Prediction of heart disease based on machine learning using jellyfish optimization algorithm,” *Diagnostics (Basel, Switzerland)*,  
   https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10378171/#::text=The%20results%20showed%20that%20RF,patients%20more%20accurately%20than%20other%20algorithms (accessed Oct. 3, 2024).

2. **World Health Organization**, “Cardiovascular diseases,” *World Health Organisation*, 2024.  
   https://www.who.int/health-topics/cardiovascular-diseases#tab=tab_1

3. **U.S. Census Bureau**, “Health and Nutrition 135 Table 205. Cumulative Percent Distribution of Population by Height and Sex: 2007 to 2008,” 2010. Available:  
   https://www2.census.gov/library/publications/2010/compendia/statab/130ed/tables/11s0205.pdf

4. **American Heart Association Obesity Committee**, “Obesity and cardiovascular disease: A scientific statement from the American Heart Association,” *Circulation, 143*(21), e984–e1010.  
   https://doi.org/10.1161/CIR.0000000000000973
""")

st.subheader("Gantt Chart")
st.markdown("""
Here is the Gantt Chart: 
<a href="https://gtvault-my.sharepoint.com/:x:/g/personal/qtran31_gatech_edu/Ef1MMvEzWS1In8petFdQO8IBkMx6Dstx9FXGdA6UWseM5w">Group 39 Gantt Chart</a>
""", unsafe_allow_html=True)

data = {
    "Name": ["Quyen Tran", "Varun Chandrashekhar", "Aryan Shah", "Aria", "Rugved", "Everyone"],
    "Proposal Contributions": [
        "Data Preprocessing",
        "Model Development/Optimization",
        "Model Development/Discussion",
        "Deployment/Discussion",
        "Model Development/Optimization",
        "Research/Model Dev"
    ]
}

df = pd.DataFrame(data)

st.subheader("Contribution Table")
st.table(df)





# Footer
st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit")
