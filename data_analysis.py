import csv
import os, sys
import json
from datetime import datetime
import re
from pytz import timezone,utc
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import math, statistics, string
import numpy as np
import heapq


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

class TopicAnalyzer():
    def __init__(self, home_folder, topic_folder, data_folder):
        self.home_folder = home_folder
        self.topic_folder = f"{self.home_folder}/ModelRes/{topic_folder}"
        self.data_folder = f"{self.home_folder}/Data4Model/{data_folder}"


    def _calculate_umass(self, word1, word2):
        w1count = 0 #total w1 in all tweets
        w2count = 0 #total w2 in all tweets
        w1w2count = 0 #total w1 w2 appearing together in all tweets

        total_tweets = 0

        for file in os.listdir(self.data_folder):
            with open(f"{self.data_folder}/{file}", "r", encoding="utf8") as reader:
                lines = reader.readlines()
                #for each tweet
                for line in lines:
                    vals = line.split(" ")

                    #if w1 exists in a tweet
                    if word1 in vals:
                        w1count += 1
                        #if w1 & w2 exists in a tweet
                        if word2 in vals:
                            w2count += 1
                            w1w2count += 1
                    #if only w2 exists
                    elif word2 in vals:
                        w2count += 1


        score = 0
        denom = max(w1count, w2count)
        if denom > 0:
            score = math.log((w1w2count + 1) / denom)

        return score

    def write_top30_words(self):
        wit_file = f"{self.topic_folder}/WordsInTopics.txt"

        writer = open(f"{self.topic_folder}/WordsInTopicsSum.txt", "w", )

        with open(wit_file, "r", encoding="utf8") as reader:
            lines = reader.readlines()



            for line in lines:
                if "Topic" in line:
                    index = 0

                if index < 30:
                    writer.write(f"{line}")
                    writer.flush()
                    index += 1


        writer.close()

    def write_theta_given_lambda(self, lam):
        vocabs, top2wrelv_matrix = self.get_word_relevance_in_topic(lam)
        topic_content = {}

        writer = open(f"{self.topic_folder}/TopicsRelevanceOnUsers.txt","w",encoding="utf8")

        with open(f"{self.topic_folder}/TopicsDistributionOnUsers.txt", "r", encoding="utf8") as reader:
            lines = reader.readlines()

            #for each file/day/document
            doc_no = 0
            for line in lines:
                topic_dist = [0] * len(top2wrelv_matrix)

                file = line.split("\t")[0].strip()
                writer.write(f"{file}\t")

                with open(f"{self.data_folder}/{file}", "r", encoding="utf8") as reader:
                    tweets = reader.readlines()

                    for tweet in tweets:
                        word_count = {}

                        words = tweet.split(" ")
                        for word in words:
                            w = word.strip()
                            word_count[w] = word_count.get(w, 0) + 1

                        vocab_indices = []
                        word_arr = []
                        word_count_arr = []
                        for word in word_count:
                            #twitter lda throw 10% of words
                            if word in vocabs:
                                vocab_indices.append(vocabs.index(word))
                                word_arr.append(word)
                                word_count_arr.append(word_count[word])

                        word_count_arr = np.array([x / sum(word_count_arr) for x in word_count_arr])

                        max_topic = -1
                        max_relevance = float("-inf")
                        for top in range(0, len(top2wrelv_matrix)):
                            wrelv = np.array(top2wrelv_matrix[top])
                            wrelv = [wrelv[i] for i in vocab_indices]

                            #indices = (-word_count_arr).argsort()[:topwords_num]
                            relv = sum([word_count_arr[i] * wrelv[i] for i in range(0, len(wrelv))])
                            if relv > max_relevance:
                                max_relevance = relv
                                max_topic = top

                        topic_content[max_topic] = topic_content.get(max_topic, set([]))
                        topic_content[max_topic].add(tweet.strip())
                        topic_dist[max_topic] += 1


                #sum of topic relevance = 1
                topic_dist = [topic_dist[i]/sum(topic_dist) for i in range(0, len(topic_dist))]


                topic_dist = [str(t) for t in topic_dist]
                writer.write("\t".join(topic_dist))
                writer.write("\n")
                writer.flush()

                doc_no += 1
                print("\r", end="")
                print("Processing", doc_no, "out of", len(lines), end="", flush=True)

        writer.close()

        print("Write topic")
        for topic in topic_content.keys():
            tweets = topic_content[topic]

            writer = open(f"{self.topic_folder}/topics/{topic}.txt","w",encoding="utf8")
            for tweet in tweets:
                writer.write(tweet)
                writer.write("\n")
                writer.flush()
            writer.close()

    def get_word_relevance_in_topic(self, lam):
        vocabs = self.write_vocab()
        top2word_matrix = self.write_phi()
        term_frequency = self.write_term_frequency()
        term_prob = [x/sum(term_frequency) for x in term_frequency]

        def calculate_relevance(p_wt,p_w):
            #print(p_w)
            relevance = lam * math.log(p_wt) + (1-lam) * math.log(p_wt/p_w)
            return relevance

        writer = open(f"{self.topic_folder}/WordsInTopics_RelvBased.txt", "w", encoding="utf8")

        top2wrelv_matrix = []

        for topic in range(0, len(top2word_matrix)):
            word_probil = top2word_matrix[topic]
            relv_probil = [calculate_relevance(word_probil[i], term_prob[i])
                           for i in range(0, len(word_probil))]
            top2wrelv_matrix.append(relv_probil)

            writer.write(f"Topic {topic}:")

            for prob,word in sorted(zip(relv_probil, vocabs), reverse=True)[0:30]:
                writer.write(f"\t{word}\t{prob}")
                writer.write("\n")
                writer.flush()

        writer.close()

        return vocabs, top2wrelv_matrix


    def _get_words_count(self):
        word_count = {}

        for file in os.listdir(self.data_folder):
            with open(f"{self.data_folder}/{file}", "r", encoding="utf8") as reader:
                lines = reader.readlines()
                for line in lines:
                    words = line.split(" ")
                    for word in words:
                        w = word.strip()
                        word_count[w] = word_count.get(w, 0) + 1

        return word_count

    def prepare_file_for_ldavis(self):
        self.write_phi()
        self.write_theta()
        self.write_doc_length()
        self.write_vocab()
        self.write_term_frequency()

    def write_term_frequency(self):
        word_file = f"{self.topic_folder}/wordMap.txt"
        term_freq = []
        word_count = self._get_words_count()

        vocab_file = f"{self.topic_folder}/term_frequency.txt"
        writer = open(vocab_file, "w", encoding="utf8")
        with open(word_file, "r", encoding="utf8") as reader:
            lines = reader.readlines()
            for line in lines:
                w = line.split("\t")
                freq = int(w[1].strip())

                freq = word_count[w[0].strip()]
                term_freq.append(freq)

                writer.write(str(freq))
                writer.write("\n")
                writer.flush()
        writer.close()

        return term_freq

    def write_vocab(self):
        vocabs = []
        word_file = f"{self.topic_folder}/wordMap.txt"

        vocab_file = f"{self.topic_folder}/vocab.txt"
        writer = open(vocab_file, "w", encoding="utf8")
        with open(word_file,"r",encoding="utf8") as reader:
            lines = reader.readlines()
            for line in lines:
                w = line.split("\t")[0].strip()
                writer.write(w)
                vocabs.append(w)
                writer.write("\n")
                writer.flush()
        writer.close()
        return vocabs


    def write_doc_length(self):
        file = f"{self.topic_folder}/TopicsDistributionOnUsers.txt"

        doc_list = []
        with open(file, "r", encoding="utf8") as reader:
            lines = reader.readlines()
            for line in lines:
                txt_index = line.index(".txt")
                doc_list.append(line[0:txt_index + 4])

        writer = open(f"{self.topic_folder}/doc_length.txt", "w", encoding="utf8")
        for d in doc_list:
            data_file = f"{self.data_folder}/{d}"
            num_token = 0
            with open(data_file, "r", encoding="utf8") as reader:
                lines = reader.readlines()
                for line in lines:
                    num_token += len(line.split(" "))
            writer.write(str(num_token))
            writer.write("\n")
            writer.flush()
        writer.close()

    def write_phi(self):
        wordmap_file = f"{self.topic_folder}/wordMap.txt"

        word_list = []
        with open(wordmap_file,"r",encoding="utf8") as reader:
            lines = reader.readlines()
            for line in lines:
                word_list.append(line.split("\t")[0].strip())

        wit_file = f"{self.topic_folder}/WordsInTopics.txt"
        topic_dist = {}

        phi_file = f"{self.topic_folder}/phi.txt"
        writer = open(phi_file,"w",encoding="utf8")

        top2word_matrix = []

        with open(wit_file, "r", encoding="utf8") as reader:
            lines = reader.readlines()

            for line in lines:
                data = line.strip().split("\t")

                if len(data) > 2:
                    if len(topic_dist) > 0:
                        #write word distribution for the current topic into file
                        total = sum(topic_dist.values())
                        top2word_array = []
                        for i in range(0, len(word_list)):
                            p = topic_dist[word_list[i]]

                            writer.write(str(p))
                            top2word_array.append(p)
                            if i < len(word_list) - 1:
                                writer.write("\t")
                            else:
                                writer.write("\n")
                        writer.flush()
                        top2word_matrix.append(top2word_array)

                    #create dictionary for a new topic
                    topic_dist = {}
                    topic_dist[data[1]] = float(data[2])
                else:
                    topic_dist[data[0]] = float(data[1])

        # write the last topic
        top2word_array = []
        for i in range(0, len(word_list)):
            p = topic_dist[word_list[i]]
            writer.write(str(p))
            top2word_array.append(p)
            if i < len(word_list) - 1:
                writer.write("\t")
            else:
                writer.write("\n")
        writer.flush()
        top2word_matrix.append(top2word_array)


        writer.close()
        return top2word_matrix


    def write_theta(self):
        file = f"{self.topic_folder}/TopicsDistributionOnUsers.txt"

        with open(file, "r", encoding="utf8") as reader:
            theta_file = f"{self.topic_folder}/theta.txt"
            writer = open(theta_file,"w",encoding="utf8")

            lines = reader.readlines()
            for line in lines:
                txt_index = line.index(".txt")
                writer.write(line[txt_index + 5:])
                writer.flush()

            writer.close()

    def count_number_of_unique_tweets(self):
        all_texts = []
        for file in os.listdir(self.data_folder):
            with open(f"{self.data_folder}/{file}", "r", encoding="utf8") as reader:
                lines = reader.readlines()
                for line in lines:
                    if line not in all_texts:
                        all_texts.append(line)
        print(len(all_texts))

    def analyze_coherence_score_of_files(self, wits_folder):
        wits_folder = f"{self.home_folder}/{wits_folder}"

        #a file represents topic-word distribution for a number of topic
        for file in os.listdir(wits_folder):
            topic_index = -1
            topic2words = {}

            full_path = f"{wits_folder}/{file}"
            with open(full_path, "r") as reader:
                lines = reader.readlines()
                for line in lines:
                    words = line.strip().split("\t")

                    if "Topic " in line:
                        topic_index += 1
                        topic2words[topic_index] = [words[1]]
                    else:
                        if len(topic2words[topic_index]) < 20:
                            topic2words[topic_index].append(words[0])

            #print(file, topic2words)
            #sys.exit()

            scores = []
            for topic in topic2words.keys():
                words = topic2words[topic]
                #print("topic",topic)

                #umass score for each topic
                total_score = 0
                for i in range(0, len(words)-1):
                    for j in range (i+1, len(words)):
                        word1 = words[i]
                        word2 = words[j]
                        total_score += self._calculate_umass(word1, word2)

                scores.append(total_score)

            mean_uci = statistics.mean(scores)
            print(f"{str(len(topic2words))},{mean_uci}")




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

    def count_geocoded_tweets(self):

        writer = open("geotweets.json", "w", encoding="utf8")

        totaltweet = 0
        sgtweet = 0
        ovstweet = 0

        with open("collected_tweets.json", "r", encoding="utf8") as reader:
            for line in reader.readlines():
                totaltweet += 1

                jtweet = json.loads(line)
                if jtweet["place"] != None:
                    if jtweet["place"]["country_code"] == "SG":
                        sgtweet += 1
                    else:
                        ovstweet += 1
                    writer.write(line)
                    writer.flush()
        writer.close()

        print("total tweet:", totaltweet)
        print("SG tweet:", sgtweet)
        print("ovs tweet:", ovstweet)


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
    def _basic_preprocess(self, text):
        #tokenize and remove stopwords

        words = word_tokenize(text)

        # step 1: remove stopwords
        stop_removed = [w for w in words if w not in set(stopwords.words('english'))]

        # step 2: remove punctuations
        punct_removed = [w for w in stop_removed if w not in string.punctuation]

        # step 3: remove non-alphabet
        regex = re.compile('[^a-zA-Z]')
        nonalph_removed = [regex.sub('', w) for w in punct_removed]

        #step 4: remove empty words
        cleaned = [w for w in nonalph_removed if w != "" and len(w) > 1]

        if len(cleaned) > 0:
            return " ".join(cleaned)
        else:
            return ""

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
                    toktweet = self._basic_preprocess(tweet)
                    if len(toktweet) > 0:
                        writer.write(f"{toktweet}\n")
                        writer.flush()
                flis_writer.write(f"{datstr}.txt\n")
                flis_writer.flush()
        flis_writer.close()

    def transform_filtered_tweets_as_raw_input(self, result_folder):
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
        flis_writer = open("filelist.txt", "w", encoding="utf8")
        for datstr in dtweet.keys():
            with open(f"{result_folder}/{datstr}.txt", "w", encoding="utf8") as writer:
                for tweet in dtweet[datstr]:
                    toktweet = " ".join(word_tokenize(tweet))
                    if len(toktweet) > 0:
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

    #sta.transform_filtered_tweets_into_csv("collected_tweets.json")
    #sta.transform_filtered_tweets_as_TLDA_input("tlda_input")
    sta.transform_filtered_tweets_as_raw_input("raw_input")
    #sta.count_geocoded_tweets()

    #sua = SeedUsersAnalyzer(folder_specs["seed users"])
    #sua.collect_tweets_with_phrase("#sgunited")

    #ta = TopicAnalyzer("C:\\Users\\fnatali\\eclipse-workspace\\Twitter-LDA-master\\data","sgu_ntopic_25", "sgunited")
    #ta.write_theta_given_lambda(0.4)
    #ta.get_word_relevance_in_topic(0.4)
    #ta.analyze_coherence_score_of_files("ModelRes/sgu")
    #ta.write_phi()
    #ta.count_number_of_unique_tweets()
    #ta.prepare_file_for_ldavis()
    #ta.write_top30_words()