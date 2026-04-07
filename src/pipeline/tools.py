# import datetime
# from langchain_core.tools import tool


# @tool("Current_Date_Time_Tool")
# def timeReciver() -> str:
#     """Essential for any query involving 'upcoming', 'next', or 'current' events. Call this FIRST."""
#     now = datetime.datetime.now()
#     return now.strftime("%d-%m-%Y %H:%M:%S")

# tools = [timeReciver] 