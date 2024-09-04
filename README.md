# Automated Data Analysis and Visualization

## Overview

This Streamlit application provides an automated solution for data analysis and visualization. It allows users to upload CSV files, perform basic data preprocessing, generate visualizations, and even train simple machine learning models.

## Features

- Data upload via CSV file
- Basic data information display
- Handling of missing values
- Data normalization and standardization
- Feature engineering (polynomial features)
- Data visualization:
  - Histograms
  - Correlation heatmaps
  - Scatter plots
- Feature importance calculation
- Machine Learning model training and evaluation
- Automated report generation

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/saksham-jain177/Automated-Data-Analysis-and-Visualization.git
   ```
2. Navigate to the project directory:
   ```
   cd Automated-Data-Analysis-and-Visualization
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit app:
``` streamlit run auto.py ```

Then, follow these steps:
1. Upload your CSV file
2. Choose options for handling missing values and data normalization
3. Explore the generated visualizations
4. Optionally, train and evaluate machine learning models
5. Download the automated report

## Dependencies

- streamlit
- pandas
- matplotlib
- seaborn
- plotly
- scikit-learn

For a complete list of dependencies, see `requirements.txt`.

## Contributing

Contributions to this project are welcome! Feel free to submit pull requests or open issues to suggest improvements or report bugs.

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgements

This project was created as a tool for automated data analysis and visualization. It's designed to be a starting point for data exploration and can be extended with additional features and capabilities.

---

Any changes and improvements are welcome! If you have ideas to enhance this tool or find any issues, please don't hesitate to contribute or reach out.
