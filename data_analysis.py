import csv
import os, sys
import json

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

    '''sta = SeedTweetsAnalyzer(folder_specs["seed tweets"])
    #requirements = {"location": "singapore", "followers": [10, float("inf")], "verified": False}
    requirements = {"protected": False}
    seed_users = sta.get_seed_users()
    filtered_users = sta.filter_seed_users(requirements)
    user_ids = [u.id for u in filtered_users]
    with open('seed_user_ids.txt', 'w') as f:
        f.write('\n'.join(user_ids))'''

    sua = SeedUsersAnalyzer(folder_specs["seed users"])
    sua.collect_tweets_with_phrase("#sgunited")
