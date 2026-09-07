"""
Export utilities for sentiment analysis results
Supports CSV, JSON, and text formats
"""
import csv
import json
import io
from typing import List, Dict
from datetime import datetime


def export_to_csv(results: List[tuple], stats: Dict = None) -> str:
    """
    Export results to CSV format.
    
    Args:
        results: List of (text, sentiment) tuples
        stats: Optional statistics dict
    
    Returns:
        CSV string
    """
    output = io.StringIO()
    writer = csv.writer(output)
    
    writer.writerow(['Text', 'Sentiment'])
    
    for text, sentiment in results:
        text_clean = text.replace('\n', ' ').replace('\r', ' ')[:1000]
        writer.writerow([text_clean, sentiment])
    
    if stats:
        writer.writerow([])
        writer.writerow(['Statistics'])
        writer.writerow(['Total', stats.get('total', 0)])
        writer.writerow(['Positive', stats.get('positive', 0)])
        writer.writerow(['Negative', stats.get('negative', 0)])
        writer.writerow(['Neutral', stats.get('neutral', 0)])
    
    return output.getvalue()


def export_to_json(results: List[tuple], stats: Dict = None, 
                   metadata: Dict = None) -> str:
    """
    Export results to JSON format.
    
    Args:
        results: List of (text, sentiment) tuples
        stats: Optional statistics dict
        metadata: Optional metadata dict
    
    Returns:
        JSON string
    """
    data = {
        "results": [
            {"text": text[:500], "sentiment": sentiment}
            for text, sentiment in results
        ],
        "stats": stats or {},
        "metadata": metadata or {},
        "exported_at": datetime.now().isoformat()
    }
    
    return json.dumps(data, indent=2)


def export_to_text(results: List[tuple], stats: Dict = None,
                   title: str = "Sentiment Analysis Results") -> str:
    """
    Export results to plain text format.
    
    Args:
        results: List of (text, sentiment) tuples
        stats: Optional statistics dict
        title: Report title
    
    Returns:
        Text string
    """
    lines = []
    lines.append("=" * 60)
    lines.append(title.center(60))
    lines.append("=" * 60)
    lines.append("")
    
    if stats:
        lines.append("SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Total:     {stats.get('total', 0)}")
        lines.append(f"Positive:  {stats.get('positive', 0)} ({stats.get('positive_pct', 0)}%)")
        lines.append(f"Negative:  {stats.get('negative', 0)} ({stats.get('negative_pct', 0)}%)")
        lines.append(f"Neutral:   {stats.get('neutral', 0)} ({stats.get('neutral_pct', 0)}%)")
        lines.append("")
    
    lines.append("DETAILED RESULTS")
    lines.append("-" * 40)
    
    for i, (text, sentiment) in enumerate(results[:100], 1):
        text_display = text[:100] + "..." if len(text) > 100 else text
        lines.append(f"{i}. [{sentiment.upper()}] {text_display}")
    
    if len(results) > 100:
        lines.append(f"\n... and {len(results) - 100} more results")
    
    lines.append("")
    lines.append(f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return "\n".join(lines)


def create_response_file(results: List[tuple], stats: Dict,
                         format: str = "csv", filename: str = None) -> tuple:
    """
    Create a downloadable file response.
    
    Returns:
        Tuple of (filename, content, mimetype)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format == "csv":
        content = export_to_csv(results, stats)
        filename = filename or f"sentiment_analysis_{timestamp}.csv"
        mimetype = "text/csv"
    
    elif format == "json":
        content = export_to_json(results, stats)
        filename = filename or f"sentiment_analysis_{timestamp}.json"
        mimetype = "application/json"
    
    elif format == "txt":
        content = export_to_text(results, stats)
        filename = filename or f"sentiment_analysis_{timestamp}.txt"
        mimetype = "text/plain"
    
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    return filename, content, mimetype
