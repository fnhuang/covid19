import csv
import os, sys
import json
from datetime import datetime
import re
from pytz import timezone,utc
from nltk import word_tokenize

FOLDER_SPEC = "folder_spec.csv"

def get_folder_spec():
    folder_specs = {}
    with open(FOLDER_SPEC,"r",encoding="utf8") as reader:
        csv_reader = csv.DictReader(reader)
        for row in csv_reader:
            folder_specs[row["description"]] = row["location"]

    return folder_specs

class TwitterUser():
    def __init__(self, id):
        self.id = id
        self.location = ""
        self.followers_count = 0
        self.verified = False
        self.protected = False

    def __eq__(self, other):
        if not isinstance(other, TwitterUser):
            # don't attempt to compare against unrelated types
            raise NotImplemented

        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

class SeedTweetsAnalyzer():
    def __init__(self, seed_tweets_folder):
        self.seed_tweets_folder = seed_tweets_folder
        self.error_instance = 0


    def _get_rt_status_if_retweeted(self, jtweet):
        is_a_retweet = False

        if "retweeted_status" in jtweet.keys():
            text = self._get_tweet_from_json(jtweet["retweeted_status"])
            is_a_retweet = True
        else:
            text = self._get_tweet_from_json(jtweet)

        return is_a_retweet,text

    def _get_tweet_from_json(self, json_obj):
        if "full_text" in json_obj.keys():
            text = json_obj["full_text"].lower()
        else:
            text = json_obj["text"].lower()

        text = re.sub(('\s\s+|\n'), ' ',text)

        return text

    def get_seed_users(self):
        seed_users = set([])

        for file in os.listdir(self.seed_tweets_folder):
            complete_path = self.seed_tweets_folder + file

            with open(complete_path, "r", encoding="utf8") as reader:
                lines = reader.readlines()

            for line in lines:
                #print(line)
                try:
                    json_user = json.loads(line)
                    user = TwitterUser(json_user["user"]["id_str"])
                    user.location = json_user["user"]["location"]
                    user.followers_count = json_user["user"]["followers_count"]
                    user.verified = json_user["user"]["verified"]
                    user.protected = json_user["user"]["protected"]

                    seed_users.add(user)
                except json.decoder.JSONDecodeError:
                    self.error_instance += 1

        return seed_users

    def filter_seed_users(self, requirements):
        seed_users = self.get_seed_users()
        filtered_users = []


        for user in seed_users:
            user_ok = True

            for key in requirements.keys():
                if key == "location":
                    if requirements[key] not in user.location.lower():
                        user_ok = False
                elif key == "followers":
                    if not (user.followers_count > requirements[key][0] and \
                            user.followers_count < requirements[key][1]):
                        user_ok = False
                elif key == "verified":
                    if user.verified != requirements[key]:
                        user_ok = False
                elif key == "protected":
                    if user.protected != requirements[key]:
                        user_ok = False

            if user_ok:
                filtered_users.append(user)


        return filtered_users

    def transform_filtered_tweets_into_csv(self, file_name):
        users = []

        tweets_file = open("tweets.csv","w",encoding="utf8",newline="")
        users_file = open("users.csv","w",encoding="utf8",newline="")

        ctwriter = csv.writer(tweets_file)
        cuwriter = csv.writer(users_file)

        ctwriter.writerow(["status_id","date_sgtime","text","is_a_retweet","retweets","likes","user_id"])
        cuwriter.writerow(["user_id","user_name","user_location","followers","followees","tweets","favourites","lists","created_at_sgtime"])

        with open("collected_tweets.json","r",encoding="utf8") as reader:
            for line in reader.readlines():
                jtweet = json.loads(line)
                is_a_retweet = False

                status_id = jtweet["id_str"]

                is_a_retweet,text = self._get_rt_status_if_retweeted(jtweet)

                created_at = datetime.strptime(jtweet["created_at"], "%a %b %d %H:%M:%S %z %Y")
                created_at = created_at.replace(tzinfo=utc).astimezone(tz=timezone("Asia/Singapore"))
                date_str = created_at.strftime("%Y-%m-%d %H:%M:%S")
                retweets = jtweet["retweet_count"]
                likes = jtweet["favorite_count"]
                user_id = jtweet["user"]["id"]

                ctwriter.writerow([status_id, date_str, text, is_a_retweet, retweets, likes, user_id])
                tweets_file.flush()

                if user_id not in users:
                    user_name = jtweet["user"]["screen_name"]
                    location = jtweet["user"]["location"] if "location" in jtweet["user"].keys() else ""
                    followers = jtweet["user"]["followers_count"]
                    friends = jtweet["user"]["friends_count"]
                    tweets = jtweet["user"]["statuses_count"]
                    favourites = jtweet["user"]["favourites_count"]
                    lists = jtweet["user"]["listed_count"]
                    created_at = datetime.strptime(jtweet["user"]["created_at"], "%a %b %d %H:%M:%S %z %Y")
                    created_at = created_at.replace(tzinfo=utc).astimezone(tz=timezone("Asia/Singapore"))
                    created_at = created_at.strftime("%Y-%m-%d %H:%M:%S")

                    cuwriter.writerow([user_id,user_name,location,followers,friends,tweets,favourites,lists,created_at])
                    users_file.flush()
                    users.append(user_id)

        tweets_file.close()
        users_file.close()


    # remove www
    def _remove_link(self,text):
        words = text.split(" ")
        valid_words = []

        for w in words:
            if not w.startswith("http") and not w.startswith("https") and not w.startswith("www"):
                valid_words.append(w)

        return " ".join(valid_words)


    # perform tokenization
    def _tokenize(self, text):
        words = word_tokenize(text)

        return " ".join(words)


    def transform_filtered_tweets_as_TLDA_input(self, result_folder):
        dtweet = {}

        # separate tweets based on dates
        with open("collected_tweets.json","r",encoding="utf8") as reader:
            for line in reader.readlines():
                jtweet = json.loads(line)

                dat = datetime.strptime(jtweet["created_at"], "%a %b %d %H:%M:%S %z %Y")
                dat = dat.replace(tzinfo=utc).astimezone(tz=timezone("Asia/Singapore"))
                dat = dat.strftime("%Y-%m-%d")

                is_a_retweet,text = self._get_rt_status_if_retweeted(jtweet)
                dtweet[dat] = dtweet.get(dat,[])
                dtweet[dat].append(text)

        # perform basic text processing, and transform into a file
        flis_writer = open("filelist.txt","w",encoding="utf8")
        for datstr in dtweet.keys():
            with open(f"{result_folder}/{datstr}.txt","w", encoding="utf8") as writer:
                for tweet in dtweet[datstr]:
                    tweet = self._remove_link(tweet)
                    toktweet = self._tokenize(tweet)
                    writer.write(f"{toktweet}\n")
                    writer.flush()
                flis_writer.write(f"{datstr}.txt\n")
                flis_writer.flush()
        flis_writer.close()



class SeedUsersAnalyzer():
    def __init__(self, seed_users_folder):
        self.seed_users_folder = seed_users_folder

    def _get_tweet_from_json(self, json_obj):
        if "full_text" in json_obj.keys():
            text = json_obj["full_text"].lower()
        else:
            text = json_obj["text"].lower()


        return text

    def collect_tweets_with_phrase(self, phrase):
        writer = open("collected_tweets.json", "w", encoding="utf8")

        for file_name in os.listdir(self.seed_users_folder):
            file_dir = f"{self.seed_users_folder}/{file_name}"

            with open(file_dir,"r",encoding="utf8") as reader:
                for line in reader.readlines():
                    json_tweet = json.loads(line)
                    if "retweeted_status" in json_tweet.keys():
                        text = self._get_tweet_from_json(json_tweet["retweeted_status"])
                    else:
                        text = self._get_tweet_from_json(json_tweet)

                    if "#sgunited" in text:
                        writer.write(f"{line}")
                        writer.flush()

        writer.close()

if __name__ == "__main__":
    folder_specs = get_folder_spec()

    sta = SeedTweetsAnalyzer(folder_specs["seed tweets"])

    #requirements = {"location": "singapore", "followers": [10, float("inf")], "verified": False}
    '''requirements = {"protected": False}
    seed_users = sta.get_seed_users()
    filtered_users = sta.filter_seed_users(requirements)
    user_ids = [u.id for u in filtered_users]
    with open('seed_user_ids.txt', 'w') as f:
        f.write('\n'.join(user_ids))'''

    sta.transform_filtered_tweets_into_csv("collected_tweets.json")

    sta.transform_filtered_tweets_as_TLDA_input("tlda_input")

    #sua = SeedUsersAnalyzer(folder_specs["seed users"])
    #sua.collect_tweets_with_phrase("#sgunited")
