import os
import re
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

def get_channel_id(query):
    #If user enters the channel ID return it directly
    if query.startswith("UC") and len(query) == 24:
        return query
        
    #Otherwise use the search endpoint to find the channel ID
    try:
        #Perform search query 
        request = youtube.search().list(
            part="snippet",
            type="channel",
            q=query,
            maxResults=1
        )
        response = request.execute()

        #Extract channel ID from the response
        if response["items"]:
            return response["items"][0]["id"]["channelId"]
        else:
            return None
    except Exception as e:
        print(f"Error finding channel ID: {e}")
        return None


def get_channel_stats(channel_id):
    #Defining that we need snippet,statistics,contentDetails from the API response
    request = youtube.channels().list(
        part="snippet,statistics,contentDetails",
        id=channel_id
    )
    response = request.execute()
    
    if not response["items"]:
        return None
    
    data = response["items"][0]
    
    stats = {
        "channel_name": data["snippet"]["title"],
        "description": data["snippet"]["description"],
        "created_at": data["snippet"]["publishedAt"],
        "thumbnail": data["snippet"]["thumbnails"]["high"]["url"],
        "subscribers": int(data["statistics"].get("subscriberCount", 0)),
        "total_views": int(data["statistics"].get("viewCount", 0)),
        "total_videos": int(data["statistics"].get("videoCount", 0)),
        "playlist_id": data["contentDetails"]["relatedPlaylists"]["uploads"]
    }
    
    return stats

def get_all_videos(playist_id):
    videos=[]
    next_page_token=None
    #Fetch all the videos from the playist pages one by one
    while True:
        request=youtube.playlistItems().list(
            part="snippet,contentDetails",
            playlistId=playist_id,
            maxResults=50,
            pageToken=next_page_token
        )
        response=request.execute()

        videos_ids=[]
        for item in response["items"]:
            videos_ids.append(item["contentDetails"]["videoId"])

        stats_request=youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=",".join(videos_ids)
        )

        stats_response=stats_request.execute()

        for video in stats_response["items"]:
            videos.append({
            "video_id": video["id"],
            "title": video["snippet"]["title"],
            "published_at": video["snippet"]["publishedAt"],
            "views": int(video["statistics"].get("viewCount", 0)),
            "likes": int(video["statistics"].get("likeCount", 0)),
            "comments": int(video["statistics"].get("commentCount", 0)),
            "duration": video["contentDetails"]["duration"]
        })
        next_page_token=response.get("nextPageToken")
        if not next_page_token:
            break
    
    return videos
            
def parse_duration(duration):
    hours = re.search(r"(\d+)H", duration)
    minutes = re.search(r"(\d+)M", duration)
    seconds = re.search(r"(\d+)S", duration)

    hours = int(hours.group(1)) if hours else 0
    minutes = int(minutes.group(1)) if minutes else 0
    seconds = int(seconds.group(1)) if seconds else 0

    return hours+ minutes+ seconds

# if __name__=="__main__":
#     channel_name="Ducky Bhai"
#     channel_id=get_channel_id(channel_name)
#     print("Channel ID:",channel_id)
#     stats=get_channel_stats(channel_id)
#     print(f"Name: {stats['channel_name']}")
#     print(f"Subscribers: {stats['subscribers']}")
#     print(f"Total Views: {stats['total_views']}")
#     print(f"Total Videos: {stats['total_videos']}")


