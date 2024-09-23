import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

class EmotionAnalysis:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.df = pd.read_csv(self.input_file)
        self.columns_to_fill = [
            'scl_avg', 'calories', 'filteredDemographicVO2Max', 'distance', 'resting_hr', 'sleep_duration',
            'minutesToFallAsleep', 'minutesAsleep', 'minutesAwake', 'minutesAfterWakeup', 'sleep_efficiency',
            'sleep_deep_ratio', 'sleep_wake_ratio', 'sleep_light_ratio', 'sleep_rem_ratio', 'steps',
            'minutes_in_default_zone_1', 'minutes_below_default_zone_1', 'minutes_in_default_zone_2',
            'minutes_in_default_zone_3', 'age', 'gender', 'bmi', 'step_goal', 'min_goal', 'max_goal',
            'step_goal_label', 'ALERT', 'HAPPY', 'NEUTRAL', 'RESTED', 'SAD', 'TENSE', 'TIRED', 'ENTERTAINMENT',
            'GYM', 'HOME', 'HOME_OFFICE', 'OTHER', 'OUTDOORS', 'TRANSIT', 'WORK'
        ]
        self.median_columns = [
            'nightly_temperature', 'spo2', 'full_sleep_breathing_rate', 'stress_score',
            'sleep_points_percentage', 'exertion_points_percentage', 'responsiveness_points_percentage',
            'daily_temperature_variation', 'bpm', 'lightly_active_minutes', 'moderately_active_minutes',
            'very_active_minutes', 'sedentary_minutes'
        ]
        self.emotion_columns = ['ALERT', 'HAPPY', 'NEUTRAL', 'RESTED', 'SAD', 'TENSE', 'TIRED']
        self.unwanted_columns = [
            'Unnamed: 0', 'id', 'date', 'badgeType', 'nremhr', 'rmssd', 'activityType', 'mindfulness_session',
            'scl_avg', 'resting_hr', 'bmi', 'step_goal', 'min_goal', 'max_goal', 'step_goal_label'
        ]
        self.svm_models = {}
        self.rf_models = {}

    def preprocess_data(self):
        print("Initial Data Overview:")
        print(self.df.describe())
        print(self.df.head(10))
        print(self.df.tail(10))
        print(self.df.isna().sum())
        print(self.df.columns)

        self.df[self.columns_to_fill] = self.df[self.columns_to_fill].fillna(0)
        self.df['activityType'] = self.df['activityType'].fillna("['NULL']")

        for column in self.median_columns:
            median_value = self.df[column].median()
            self.df[column] = self.df[column].fillna(median_value)

        self.df.to_csv(self.output_file, index=False)
        print(self.df.isna().sum())

    def analyze_emotions(self):
        emotion_data = self.df[self.emotion_columns].sum()
        emotion_data.plot(kind='bar')
        plt.title('Total Counts of Emotions')
        plt.xlabel('Emotion')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('emotion.jpg')

    def prepare_data(self):
        unwanted_columns = [col for col in self.unwanted_columns if col in self.df.columns]
        df_model = self.df.drop(columns=unwanted_columns)

        label_encoder = LabelEncoder()
        for column in df_model.select_dtypes(include=['object']).columns:
            df_model[column] = label_encoder.fit_transform(df_model[column].astype(str))

        X = df_model.drop(columns=self.emotion_columns)
        y = df_model[self.emotion_columns]
        print(X.isna().sum())
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_models(self, X_train, X_test, y_train, y_test):
        for emotion in self.emotion_columns:
            print(f"Training SVM Model for {emotion}")
            svm_model = SVC()
            svm_model.fit(X_train, y_train[emotion])
            self.svm_models[emotion] = svm_model

            y_pred = svm_model.predict(X_test)
            cm = confusion_matrix(y_test[emotion], y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
            disp.plot(cmap=plt.cm.Blues)
            plt.title(f'SVM Confusion Matrix for {emotion}')
            plt.savefig(f'{emotion}_SVM.jpg')
            print(f'SVM Model for {emotion}')
            print(f'Accuracy: {accuracy_score(y_test[emotion], y_pred)}')
            print(classification_report(y_test[emotion], y_pred))

            print(f"Training Random Forest Model for {emotion}")
            rf_model = RandomForestClassifier()
            rf_model.fit(X_train, y_train[emotion])
            self.rf_models[emotion] = rf_model

            y_pred_rf = rf_model.predict(X_test)
            cm = confusion_matrix(y_test[emotion], y_pred_rf)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
            disp.plot(cmap=plt.cm.Blues)
            plt.title(f'Random Forest Confusion Matrix for {emotion}')
            plt.savefig(f'{emotion}_Random_forest.jpg')
            print(f'Random Forest Model for {emotion}')
            print(f'Accuracy: {accuracy_score(y_test[emotion], y_pred_rf)}')
            print(classification_report(y_test[emotion], y_pred_rf))

if __name__ == "__main__":
    analysis = EmotionAnalysis('daily_fitbit_sema_df_unprocessed.csv', 'data_updated.csv')
    analysis.preprocess_data()
    analysis.analyze_emotions()
    X_train, X_test, y_train, y_test = analysis.prepare_data()
    analysis.train_models(X_train, X_test, y_train, y_test)
