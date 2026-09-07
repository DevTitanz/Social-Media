
import os
import re
import logging
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import config

logger = logging.getLogger(__name__)

YOUTUBE_API_KEY = config.YOUTUBE_API_KEY

class YouTubeAPIError(Exception):
    pass

class InvalidVideoURLError(Exception):
    pass

def validate_video_url(url):
    if not url or not isinstance(url, str):
        raise InvalidVideoURLError("Video URL is required")
    
    url = url.strip()
    if not url:
        raise InvalidVideoURLError("Video URL cannot be empty")
    
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11})"
    if not re.search(pattern, url):
        raise InvalidVideoURLError("Invalid YouTube URL format")
    
    return True

def extract_video_id(url):
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    return None


def fetch_youtube_comments(video_url, max_results):
    validate_video_url(video_url)
    
    if not YOUTUBE_API_KEY:
        raise YouTubeAPIError(
            "YouTube API key not configured. "
            "Please set YOUTUBE_API_KEY in .env file. "
            "Copy .env.example to .env and add your API key."
        )
    
    if not (1 <= max_results <= config.MAX_YOUTUBE_COMMENTS):
        raise ValueError(f"Number of comments must be between 1 and {config.MAX_YOUTUBE_COMMENTS}")
    
    video_id = extract_video_id(video_url)
    
    if not video_id:
        raise InvalidVideoURLError("Could not extract video ID from URL")
    
    try:
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_results
        )
        
        response = request.execute()
        
    except HttpError as e:
        error_reason = e.error_details[0].get('reason', 'unknown') if e.error_details else str(e)
        
        if error_reason == 'videoNotFound':
            raise YouTubeAPIError("Video not found. Please check the URL.")
        elif error_reason == 'commentsDisabled':
            raise YouTubeAPIError("Comments are disabled for this video.")
        elif error_reason == 'quotaExceeded':
            raise YouTubeAPIError("API quota exceeded. Please try again later.")
        else:
            logger.error(f"YouTube API error: {error_reason}")
            raise YouTubeAPIError(f"Failed to fetch comments: {error_reason}")
    
    except Exception as e:
        logger.error(f"Unexpected error fetching YouTube comments: {e}")
        raise YouTubeAPIError(f"Failed to connect to YouTube API: {str(e)}")
    
    comments = []
    
    if "items" in response:
        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
    
    return comments
