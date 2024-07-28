from db_reader import TweeterMongoDB, DataSource
import datetime

def prompt_test():
    
    from tweet_classifier import ClassificationTask, TaskName
    task = ClassificationTask(TaskName.IsraelSentimentTweet)
    print (task.prompt_template(["hello", "world"]))

if __name__ == "__main__":
    prompt_test()


