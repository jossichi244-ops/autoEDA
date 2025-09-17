# tests/test_full_eda.py
import pytest
from ..main import (
    inspect_dataset,
    clean_dataset,
    descriptive_statistics,
    generate_visualizations,
    generate_relationships,
    generate_advanced_eda,
    extract_eda_insights
)
from test_dataset import GOLDEN_DF

def test_full_eda_pipeline():
    """Test toàn bộ pipeline EDA với golden dataset"""
    
    # Step 1: Inspection
    inspection = inspect_dataset(GOLDEN_DF)
    assert inspection is not None
    assert "missing_summary" in inspection
    assert "duplicates" in inspection
    assert inspection["duplicates"]["duplicate_count"] > 0

    # Step 2: Cleaning
    cleaned = clean_dataset(GOLDEN_DF, multi_select_cols=["skills"])
    assert cleaned is not None
    assert "cleaned_preview" in cleaned
    assert "summary" in cleaned

    # Step 3: Descriptive Statistics
    descriptive = descriptive_statistics(GOLDEN_DF)
    assert descriptive is not None
    assert "numeric" in descriptive
    assert "categorical" in descriptive
    assert len(descriptive["remarks"]) > 0

    # ✅ Step 3.5: Test generate_visualizations 
    visualizations = generate_visualizations(GOLDEN_DF)
    assert visualizations is not None
    assert isinstance(visualizations, dict)
    assert "numeric" in visualizations
    assert "categorical" in visualizations
    # Kiểm tra ít nhất 1 biểu đồ numeric
    if len(descriptive["numeric"]) > 0:
        first_num_col = list(descriptive["numeric"].keys())[0]
        assert first_num_col in visualizations["numeric"]
        assert "histogram" in visualizations["numeric"][first_num_col]

    # ✅ Step 3.6: Test generate_relationships
    relationships = generate_relationships(GOLDEN_DF)
    assert relationships is not None
    assert isinstance(relationships, dict)
    assert "categorical_vs_categorical" in relationships
    assert "numeric_vs_numeric" in relationships
    assert "mixed" in relationships
    assert "interactive" in relationships

    # Step 4: Advanced EDA 
    advanced = generate_advanced_eda(GOLDEN_DF)
    assert advanced is not None
    assert "clustering" in advanced
    assert "significance" in advanced
    assert "patterns" in advanced
    assert "redundancy" in advanced
    assert "timeseries" in advanced

    # Step 5: Insights
    mock_result = {
        "inspection": inspection,
        "descriptive": descriptive,
        "visualizations": visualizations,      
        "relationships": relationships,       
        "advanced": advanced
    }
    insights = extract_eda_insights(mock_result)
    assert isinstance(insights, list)
    assert len(insights) > 0
    assert any("duplicate" in insight.lower() for insight in insights)
    assert any("missing" in insight.lower() for insight in insights)
    assert any("outlier" in insight.lower() for insight in insights)

    print(f"✅ Generated {len(insights)} insights successfully!")