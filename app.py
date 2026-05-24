{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMJGWMjnI9lpZNcDDgkobd6",
      "include_colab_link": True
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Garu84/Iris_dashboard/blob/main/app.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 383
        },
        "id": "fiNxumJzUPmB",
        "outputId": "c84c6cf4-04c8-4f35-d00c-1ba5537be7ef"
      },
      "outputs": [
        {
          "output_type": "error",
          "ename": "ModuleNotFoundError",
          "evalue": "No module named 'streamlit'",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
            "\u001b[0;32m/tmp/ipykernel_1363/2475539771.py\u001b[0m in \u001b[0;36m<cell line: 0>\u001b[0;34m()\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0;32mimport\u001b[0m \u001b[0mstreamlit\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mst\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      2\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mpandas\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mpd\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mmatplotlib\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpyplot\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mplt\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mseaborn\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0msns\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mModuleNotFoundError\u001b[0m: No module named 'streamlit'",
            "",
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0;32m\nNOTE: If your import is failing due to a missing package, you can\nmanually install dependencies using either !pip or !apt.\n\nTo view examples of installing some common dependencies, click the\n\"Open Examples\" button below.\n\u001b[0;31m---------------------------------------------------------------------------\u001b[0m\n"
          ],
          "errorDetails": {
            "actions": [
              {
                "action": "open_url",
                "actionText": "Open Examples",
                "url": "/notebooks/snippets/importing_libraries.ipynb"
              }
            ]
          }
        }
      ],
      "source": [
        "import streamlit as st\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.metrics import accuracy_score\n",
        "from sklearn.metrics import precision_score\n",
        "from sklearn.metrics import recall_score\n",
        "from sklearn.metrics import f1_score\n",
        "\n",
        "# PAGE CONFIGURATION\n",
        "st.set_page_config(page_title=\"Iris Species Classification\",layout=\"wide\")\n",
        "\n",
        "# LOAD DATA\n",
        "@st.cache_data\n",
        "def load_data():\n",
        "    return pd.read_csv(\"Iris.csv\")\n",
        "\n",
        "df = load_data()\n",
        "\n",
        "# TITLE\n",
        "st.title(\"Iris Species Classification Dashboard\")\n",
        "st.write(\"Machine Learning dashboard for predicting Iris flower species.\")\n",
        "\n",
        "# PREPARE DATA\n",
        "X = df.drop([\"Species\", \"Id\"], axis=1)\n",
        "y = df[\"Species\"]\n",
        "\n",
        "# SPLIT DATA\n",
        "\n",
        "X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)\n",
        "\n",
        "# TRAIN MODEL\n",
        "model = RandomForestClassifier()\n",
        "model.fit(X_train, y_train)\n",
        "\n",
        "# PREDICTIONS\n",
        "y_pred = model.predict(X_test)\n",
        "\n",
        "# METRICS\n",
        "accuracy = accuracy_score(y_test, y_pred)\n",
        "precision = precision_score(y_test,y_pred,average=\"weighted\")\n",
        "recall = recall_score(y_test,y_pred,average=\"weighted\")\n",
        "\n",
        "f1 = f1_score(y_test,y_pred,average=\"weighted\")\n",
        "\n",
        "# METRIC CARDS\n",
        "col1, col2, col3, col4 = st.columns(4)\n",
        "col1.metric(\"Accuracy\", round(accuracy, 2))\n",
        "col2.metric(\"Precision\", round(precision, 2))\n",
        "col3.metric(\"Recall\", round(recall, 2))\n",
        "col4.metric(\"F1 Score\", round(f1, 2))\n",
        "\n",
        "# SIDEBAR INPUTS\n",
        "st.sidebar.header(\"Flower Measurements\")\n",
        "sepal_length = st.sidebar.slider(\"Sepal Length\",4.0,8.0,5.1)\n",
        "sepal_width = st.sidebar.slider(\"Sepal Width\",2.0,4.5,3.5)\n",
        "petal_length = st.sidebar.slider(\"Petal Length\",1.0,7.0,1.4)\n",
        "petal_width = st.sidebar.slider(\"Petal Width\",0.1,2.5,0.2)\n",
        "\n",
        "# USER INPUT DATA\n",
        "input_data = pd.DataFrame({\n",
        "    \"SepalLengthCm\": [sepal_length],\n",
        "    \"SepalWidthCm\": [sepal_width],\n",
        "    \"PetalLengthCm\": [petal_length],\n",
        "    \"PetalWidthCm\": [petal_width]})\n",
        "\n",
        "# PREDICTION\n",
        "prediction = model.predict(input_data)\n",
        "\n",
        "# SHOW PREDICTION\n",
        "st.subheader(\"Predicted Species\")\n",
        "st.success(prediction[0])\n",
        "\n",
        "# SPECIES DISTRIBUTION\n",
        "st.subheader(\"Species Distribution\")\n",
        "fig1, ax1 = plt.subplots(figsize=(8, 5))\n",
        "sns.countplot(data=df,x=\"Species\",ax=ax1)\n",
        "ax1.set_title(\"Species Distribution\")\n",
        "st.pyplot(fig1)\n",
        "\n",
        "# SCATTERPLOT\n",
        "st.subheader(\"Petal Length vs Petal Width\")\n",
        "fig2, ax2 = plt.subplots(figsize=(8, 5))\n",
        "sns.scatterplot(data=df,x=\"PetalLengthCm\",y=\"PetalWidthCm\",hue=\"Species\",ax=ax2)\n",
        "ax2.set_title(\"Petal Length vs Petal Width\")\n",
        "st.pyplot(fig2)\n",
        "\n",
        "# DATASET PREVIEW\n",
        "st.subheader(\"Dataset Preview\")\n",
        "st.dataframe(df)"
      ]
    }
  ]
}
