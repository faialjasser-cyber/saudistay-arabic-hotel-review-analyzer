# SaudiStay — Arabic Hotel Review Analyzer

SaudiStay is a beginner-friendly Arabic NLP project for analyzing reviews of hotels located in Saudi Arabia. It includes model training and a Streamlit web application.

## App features

- Predict sentiment for one Arabic review.
- Upload CSV or Excel files of hotel reviews.
- Filter uploaded reviews by Saudi hotel location.
- View sentiment charts by location or hotel and download the analyzed results.

## Data source and limitation

The starter uses the balanced Arabic hotel-review workbook linked from the [LSAHR dataset repository](https://github.com/mnskaho/LSAHR_dataset). The workbook is a mixed-country dataset with hotel names, location information, Arabic review text, ratings, and polarity scores. SaudiStay filters by the hotel's location. Verify the Saudi row count and locations printed during training before relying on results. The field names can vary, so the app lets you select the review and location columns.

A 2020 study reports 4,604 Arabic reviews for 121 hotels in Saudi Arabia, but its article does not provide a separate downloadable data file. This starter does not claim to include that study's data.

Do not commit the full workbook to a public GitHub repository unless its data license and source terms clearly allow redistribution. The workbook is excluded by .gitignore.

## Run in Colab or locally

1. Download the balanced workbook from the dataset repository above.
2. Place it in the data folder as balanced_reviews.xlsx.
3. Install requirements with: pip install -r requirements.txt
4. Train the model with: python train_model.py --data data/balanced_reviews.xlsx
5. Review the printed Saudi review count, city/location values, class distribution, accuracy, classification report, and confusion matrix.
6. Start the app with: streamlit run app.py

The training script saves models/saudi_hotel_sentiment.joblib. Commit the trained model if you want the deployed app to make predictions. Keep the source workbook out of GitHub.

## Model

The baseline combines word and character TF-IDF features with Logistic Regression. Character n-grams can help with spelling and dialect differences. Labels use the dataset's polarity_score column when available. Otherwise, numeric ratings are mapped as 1–4 negative, 5–6 neutral, and 7–10 positive; rating-derived labels are approximate.

## Deploy

Push the source files and trained model to GitHub. On Streamlit Community Cloud, connect the repository and choose app.py as the entrypoint.

## Suggested repository name

saudistay-arabic-hotel-review-analyzer

## Important limitation

The Saudi subset may be much smaller than the full workbook. Report the actual filtered row count and evaluate on a held-out Saudi subset. Results may not generalize to every city, hotel, or Arabic dialect.
