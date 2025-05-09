# OpenFDA API Dashboard

## Project Summary

This dashboard application provides interactive visualizations and AI-powered insights for FDA data across multiple regulatory domains.

### Development Process

1. **Initial Implementation**:
   - Created a comprehensive data infrastructure with caching and multithreading
   - Developed dedicated API modules for Drug, Food, Tobacco, Device, and other endpoints
   - Built interactive UI with visualizations and local filtering capabilities

2. **AI Integration**:
   - Implemented Gemini AI-powered trend analysis for healthcare data
   - Added predictive analytics for drug safety, food safety, and tobacco health effects
   - Enabled natural language querying of FDA data patterns

3. **Refinement**:
   - Resolved component duplication issues across pages
   - Streamlined codebase by removing unused modules
   - Optimized data loading with concurrent processing

### Key Findings

- The FDA API provides rich, accessible data across multiple regulatory domains
- AI integration significantly enhances data interpretation capabilities
- Multi-category analysis reveals cross-domain health trends that would be difficult to identify manually
- Concurrent data fetching dramatically improves application performance

### Technologies

- **Backend**: Python with OpenFDA API integration
- **Frontend**: Streamlit for interactive visualizations
- **AI**: Google Gemini for trend analysis and predictions
- **Data Processing**: Pandas, NumPy, and concurrent.futures

## Installation and Usage

### Prerequisites
- Python 3.8+
- Git

### Installation
1. Clone the repository:
   ```
   git clone https://github.com/MosetiObadiah/openFDA-Data-Visualization.git
   cd openFDA-Data-Visualization
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with your API keys:
   ```
   OPENFDA_API_KEY=your_openFDA_api_key
   GEMINI_API_KEY=your_gemini_api_key
   ```

### Running the Application
1. Start the Streamlit server:
   ```
   streamlit run app/main.py
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:8501
   ```

3. Navigate through the different sections using the sidebar to explore FDA data across various categories.

### Features
- **Interactive Dashboards**: Visualize FDA data with dynamic filtering
- **AI Insights**: Get AI-powered analysis of healthcare trends
- **Predictive Analytics**: Ask questions about future trends based on FDA data patterns
- **Multi-domain Analysis**: Explore drug, food, tobacco, and device safety data
