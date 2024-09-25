# -*- coding: utf-8 -*-

import pandas as pd
import argparse


'''change the format of the prompt for different LLMs'''
llama_format = "<s>[INST]<<SYS>>You are a helpful assistant.<</SYS>>{user_input}[/INST]"
gemma_format = "<start_of_turn>user\n{user_input}<end_of_turn><start_of_turn>model\n"
qwen_format = "<|im_start|>user\n{user_input}|im_end|>\n<|im_start|>assistant\n"
openchat_format = "GPT4 Correct User: {user_input}\n<|end_of_turn|>GPT4 Correct Assistant:"
mixtral_format = "<s>[INST]{user_input}[/INST]"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate prompts for vLLM evaluation")
    parser.add_argument("--prompt_format", type=str, default=llama_format, help="The format of the prompt for different LLMs")
    parser.add_argument("--main_file", type=str, default=None, help="The main file (path) containing source, reference, machine translation and DA scores")
    args = parser.parse_args()
    return args

class VllmTemplate:
    '''
    main_file: the main file (path) containing source, reference, machine translation and DA scores
    error_file: the file (path) containing erroneous words, if none, error words will be taken from the main file
    previous_output_file: A list of filepath for the output from previous prompts IN ORDER. It's mandatory for the SECOND prompt of Template 7, but not for all other prompt templates.
    format: the format of the prompt for different LLMs, default is llama_format in parse_args()
    '''
    def __init__(self, main_file, error_file=None, previous_output_file=None, format=None) -> None:
        self.lang_pair_short = main_file.split("/")[-1].split("_")[0]
        pair = {
            "en-de": "English-German",
            "en-mr": "English-Marathi",
            "en-zh": "English-Chinese",
            "et-en": "Estonian-English",
            "ne-en": "Nepali-English",
            "ro-en": "Romanian-English",
            "ru-en": "Russian-English",
            "si-en": "Sinhala-English",
        }
        self.lang_pair = pair[self.lang_pair_short]
        self.main_file = pd.read_csv(main_file, sep="\t", encoding="utf-8")
        if error_file:
            with open(error_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
        else:
            lines = self.main_file["error_words"].tolist()
        self.error_file = [str(line).strip() for line in lines]
        if previous_output_file:
            if previous_output_file[0].endswith(".tsv"):
                self.prompt1_output = pd.read_csv(previous_output_file[0], sep="\t", encoding="utf-8")["vllm_output"].tolist()
            elif previous_output_file[0].endswith(".csv"):
                self.prompt1_output = pd.read_csv(previous_output_file[0], encoding="utf-8")["prediction"].tolist()
            else:
                raise ValueError("Invalid file format!")

        if format:
            self.format = format
            

    def generate_template01(self):
        '''template01 source + MT'''
        instruction_list = []
        source = self.lang_pair.split("-")[0]
        target = self.lang_pair.split("-")[1]
        for i in range(len(self.main_file)):
            instruction = "Score the following translation from %s to %s by comparing the source and the translation on a continuous scale from 0 to 100, where score of zero means \"no meaning preserved\" and score of one hundred means \"perfect meaning and grammar\".\n%s source: %s\n%s translation: %s\nScore: " % (
                source,
                target,
                source,
                self.main_file["src"][i],
                target,
                self.main_file["mt"][i],
            )
            instruction = self.format.format(user_input=instruction)
            instruction_list.append(instruction)

        df = pd.DataFrame({"final_prompt": instruction_list})
        df.to_csv(self.lang_pair_short + "_vllm_t01.tsv", index=True, encoding="utf-8", sep="\t")


    def generate_template02(self):
        '''template02 MT + REF'''
        instruction_list = []
        source = self.lang_pair.split("-")[0]
        target = self.lang_pair.split("-")[1]
        for i in range(len(self.main_file)):
            instruction = "Score the following translation from %s to %s with respect to the human reference on a continuous scale from 0 to 100, where score of zero means \"no meaning preserved\" and score of one hundred means \"perfect meaning and grammar\".\n%s translation: %s\n%s human reference: %s\nScore: " % (
                source,
                target,
                target,
                self.main_file["mt"][i],
                target,
                self.main_file["ref"][i],
            )
            instruction = self.format.format(user_input=instruction)
            instruction_list.append(instruction)

        df = pd.DataFrame({"final_prompt": instruction_list})
        df.to_csv(self.lang_pair_short + "_vllm_t02.tsv", index=True, encoding="utf-8", sep="\t")


    def generate_template03(self):
        '''template03 source + REF + MT'''
        instruction_list = []
        source = self.lang_pair.split("-")[0]
        target = self.lang_pair.split("-")[1]
        for i in range(len(self.main_file)):
            instruction = "Score the following translation from %s to %s with respect to the human reference on a continuous scale from 0 to 100, where score of zero means \"no meaning preserved\" and score of one hundred means \"perfect meaning and grammar\". \n%s source: %s\n%s human reference: %s\n%s translation: %s\nScore: " % (
                source,
                target,
                source,
                self.main_file["src"][i],
                target,
                self.main_file["ref"][i],
                target,
                self.main_file["mt"][i],
            )
            instruction = self.format.format(user_input=instruction)
            instruction_list.append(instruction)

        df = pd.DataFrame({"final_prompt": instruction_list})
        df.to_csv(self.lang_pair_short + "_vllm_t03.tsv", index=True, encoding="utf-8", sep="\t")


    def generate_template04(self):
        '''template04 source + MT + error words'''
        instruction_list = []
        source = self.lang_pair.split("-")[0]
        target = self.lang_pair.split("-")[1]
        for i in range(len(self.main_file)):
            instruction = "Score the following translation from %s to %s by comparing the source and the translation and considering the error words in translation on a continuous scale from 0 to 100, where score of zero means \"no meaning preserved\" and score of one hundred means \"perfect meaning and grammar\". \n%s source: %s\n%s translation: %s\nError words: %s\nScore: " % (
                source,
                target,
                source,
                self.main_file["src"][i],
                target,
                self.main_file["mt"][i],
                self.main_file["error_words"][i] if isinstance(self.main_file["error_words"][i], str) else "",
            )
            instruction = self.format.format(user_input=instruction)
            instruction_list.append(instruction)

        df = pd.DataFrame({"final_prompt": instruction_list})
        df.to_csv(self.lang_pair_short + "_vllm_t04.tsv", index=True, encoding="utf-8", sep="\t")


    def generate_template05(self):
        '''template05 source + REF + MT + error words'''
        instruction_list = []
        source = self.lang_pair.split("-")[0]
        target = self.lang_pair.split("-")[1]
        for i in range(len(self.main_file)):
            instruction = "Score the following translation from %s to %s with respect to the human reference and considering the error words in translation on a continuous scale from 0 to 100, where score of zero means \"no meaning preserved\" and score of one hundred means \"perfect meaning and grammar\". \n%s source: %s\n%s human reference: %s\n%s translation: %s\nError words: %s\nScore: " % (
                source,
                target,
                source,
                self.main_file["src"][i],
                target,
                self.main_file["ref"][i],
                target,
                self.main_file["mt"][i],
                self.main_file["error_words"][i] if isinstance(self.main_file["error_words"][i], str) else "",
            )
            instruction = self.format.format(user_input=instruction)
            instruction_list.append(instruction)

        df = pd.DataFrame({"final_prompt": instruction_list})
        df.to_csv(self.lang_pair_short + "_vllm_t05.tsv", index=True, encoding="utf-8", sep="\t")


    def generate_template06(self):
        '''template06 source + REF + MT + annotation guidelines'''
        instruction_list = []
        source = self.lang_pair.split("-")[0]
        target = self.lang_pair.split("-")[1]
        for i in range(len(self.main_file)):
            instruction = "Score the following translation from %s to %s with respect to the human reference on a continuous scale from 0 to 100, where score of zero means \"no meaning preserved\" in terms of accuracy, contextual understanding, grammar, syntax, overall readability and score of one hundred means \"perfect meaning and grammar\" in terms of accuracy, contextual understanding, grammar, syntax, overall readability.\n%s source: %s\n%s human reference: %s\n%s translation: %s\nScore: " % (
                source,
                target,
                source,
                self.main_file["src"][i],
                target,
                self.main_file["ref"][i],
                target,
                self.main_file["mt"][i],
            )
            instruction = self.format.format(user_input=instruction)
            instruction_list.append(instruction)

        df = pd.DataFrame({"final_prompt": instruction_list})
        df.to_csv(self.lang_pair_short + "_vllm_t06.tsv", index=True, encoding="utf-8", sep="\t")


    def generate_CoT_prompt1(self, for_history=False):
        '''The first prompt in CoT'''
        instruction_list = []
        for i in range(len(self.main_file)):
            instruction = "You are going to evaluate the quality for %s translation. You need to think step by step. First read the following source, machine translation and reference translation. Analyze where the machine translation is different from the reference translation.\nSource: %s\nMachine translation: %s\nReference translation: %s" % (
                self.lang_pair,
                self.main_file["src"][i],
                self.main_file["mt"][i],
                self.main_file["ref"][i],
                )

            instruction = self.format.format(user_input=instruction)
            instruction_list.append(instruction)

        if for_history:
            return instruction_list
        else:
            df = pd.DataFrame({"final_prompt": instruction_list})
            df.to_csv(self.lang_pair_short + "_vllm_t7p1.tsv", index=True, encoding="utf-8", sep="\t")

    def generate_CoT_prompt2(self, for_history=False):
        '''The second prompt in CoT'''
        instruction_list = []
        for i in range(len(self.main_file)):
            instruction = "A large language model did an evaluation of the %s translation, which is given as below:\n%s\nBased on above analysis, score the machine translation quality on a continuous scale from 0 to 100, where score of zero means \"no meaning preserved\" and score of one hundred means \"perfect meaning and grammar\". Provide the score strictly in JSON format. " % (
                self.lang_pair,
                self.prompt1_output[i],
                )
            instruction = self.format.format(user_input=instruction)
            instruction_list.append(instruction)

        if for_history:
            return instruction_list
        else:
            df = pd.DataFrame({"final_prompt": instruction_list})
            df.to_csv(self.lang_pair_short + "_vllm_t7p2.tsv", index=True, encoding="utf-8", sep="\t")


    def generate_template08(self):
        '''template08 for few-shot learning'''
        source = self.lang_pair.split("-")[0]
        target = self.lang_pair.split("-")[1]
        instruction_list = []
        if self.lang_pair_short == "en-zh":
            examples = "Example 1\nSource: Strange; Lieutenant Colonels John T. Ellis, Charles S. Peyton, and Bennett Taylor; and Majors Waller M. Boyd and William Watts.\nMachine translation: 奇怪的是 ， John T. Ellis 中校、 Charles S. Peyton 和 Bennett Taylor 以及 Waller M. Boyd 和 William Watts 少校。\nReference translation: 奇怪；JohnT.Ellis中校、CharlesS.Peyton和BennettTaylor以及WallerM.Boyd和WilliamWatts少校。\nScore: 16\n\nExample 2\nSource: Then in 2005, AFI ranked John \"Bluto\" Blutarsky's quote \"Toga!\nMachine translation: 然后在 2005 年 ， AFI 给约翰 \"蓝天\" 的 \"瑜伽\" 排名 ！\nReference translation: 然后在2005年，AFI将约翰的“Bluto”Blutarsky名言“Toga！\nScore: 33\n\nExample 3\nSource: He now owned an Iași townhouse and a villa in Bucharest's Filipescu Park.\nMachine translation: 他现在在布加勒斯特的菲律宾人公园拥有一座雅希镇别墅和一座别墅。\nReference translation: 他现在在布加勒斯特的菲律宾人公园拥有一座Iași联排别墅和一座别墅。\nScore: 45\n\nExample 4\nSource: Plants inhabiting the watershed include conifer and hardwood trees, herbs, legumes, and grasses.\nMachine translation: 居住在流域的植物包括针叶树和硬木树、草药、豆类和草。\nReference translation: 居住在该流域的植物包括针叶树和阔叶树、草药、豆类和草。\nScore: 69\n\nExample 5\nSource: It melts at 221 °C to a black liquid and boils at 685 °C to a dark yellow vapour.\nMachine translation: 它在 221 摄氏度下熔化为黑色液体 ， 在 685 摄氏度下沸腾为深黄色蒸气。\nReference translation: 它在221°C熔化为黑色液体，在685°C沸腾为深黄色蒸气。\nScore: 84"

        elif self.lang_pair_short == "en-mr":
            examples = "Example 1\nSource: Two cotton knots attached to strike the heads alternately when manipulated.\nMachine translation: दोनदा दोनदा एकापाठोपाठ एक कांद्याचे दोन ठोके जोडून ते डोक्यावर मारले जातात.\nReference translation: फेरफार केल्यावर डोक्यावर आळीपाळीने वार करण्यासाठी कापसाच्या दोन गाठी जोडल्या जातात . \nScore: 14\n\nExample 2\nSource: He bends forward exerting the full weight of his body which is emphasized by the powerful shoulders and the bold diagonal movement of the left hand across the chest.\nMachine translation: तो आपल्या शरीराचे पूर्ण वजन करून पुढे झुकतो ज्यावर उजव्या हाताची छाती आणि उजव्या हाताची कर्कश कर्कश हालचाल अवलंबून असते.\nReference translation: तो आपल्या शरीराचे पूर्ण वजन करून पुढे झुकतो ज्यावर ताकदवान खांदे आणि छातीवर डाव्या हाताची ठळक कर्णरेषा यांद्वारे जोर दिला जातो . \nScore: 35\n\nExample 3\nSource: You should be able to wash your hair after about a week, providing you do not get water inside your ear.\nMachine translation: सुमारे एक आठवड्यानंतर आपले केस धुणे शक्य व्हावे, जेणेकरून आपल्या कानात पाणी नसेल. \nReference translation: सुमारे एक आठवड्यानंतर तुम्हाला केस धुणे शक्य आहे , जेणेकरून तुमच्या कानात पाणी जाणार नाही . \nScore: 60\n\nExample 4\nSource: Dial 999 immediately to request an ambulance if you're with someone who experiences any of these symptoms after a head injury.\nMachine translation: डोक्याला दुखापत झाल्यानंतर यापैकी कोणतीही लक्षणे आढळल्यास त्वरित रुग्णवाहिकेसाठी 999 डायल करा.\nReference translation: डोक्याला दुखापत झाल्यानंतर यापैकी कोणतीही लक्षणे अनुभवणाऱ्या व्यक्तीसोबत तुम्ही असाल तर त्वरित रुग्णवाहिकेसाठी ९९९ डायल करा .\nScore: 76\n\nExample 5\nSource: Like the One Nation and the One Mobility Card, our government has done many things in the last few years to consolidate systems of the country.\nMachine translation: 'वन नेशन' आणि 'वन मोबिलिटी कार्ड' प्रमाणेच आमच्या सरकारने गेल्या काही वर्षात देशातील व्यवस्था मजबूत करण्यासाठी खूप काही केले आहे.\nReference translation: ' वन नेशन ' आणि ' वन मोबिलिटी कार्ड ' प्रमाणेच आमच्या सरकारने गेल्या काही वर्षात देशातील व्यवस्था मजबूत करण्यासाठी खूप काही केले आहे . \nScore: 81"

        elif self.lang_pair_short == "en-de":
            examples = "Example 1\nSource: In his last three seasons, McCabe took only one wicket, from 18 overs of bowling.\nMachine translation: In seinen letzten drei Jahreszeiten, McCabe nahm nur einen Wicker, von 18 overs Bowling.\nReference translation: In seinen letzten drei Saisonen nahm McCabe nur eine Pforte und das von 18 Overs Bowling.\nScore: 16\n\nExample 2\nSource: The church and chapel fielded useful sides, also very often second elevens.\nMachine translation: Die Kirche und Kapelle nutzten Seiten, auch sehr oft zweite erhebt.\nReference translation: Die Kirche und Kapelle schickten gute Seiten aufs Feld , sehr oft auch zweite elfs.\nScore: 37\n\nExample 3\nSource: In his single semester there, he averaged 17.6 points and 13.3 rebounds, before flunking out due to poor academic performance.\nMachine translation: In seinem Semester dort lag er im Durchschnitt bei 17,6 Punkten und 13,3 Rebounds, bevor er aufgrund schlechter akademischer Leistungen ausflog.\nReference translation: In seinem einzigen Semester dort lag er im Durchschnitt bei 17,6 Punkten und 13,3 Rebounds , bevor er aufgrund schlechter akademischer Leistungen von der Schule flog .\nScore: 55\n\nExample 4\nSource: Obturating rings for the guns ran out, and the LAD had to improvise them from suet.\nMachine translation: Obturierringe für die Gewehre liefen aus, und die LAD musste sie aus dem Suet improvisieren.\nReference translation: Die Obturationsringe für die Waffen gingen aus , und die LAD musste sie aus dem Suet improvisieren .\nScore: 65\n\nExample 5\nSource: All our own guns were either smashed or dismounted by the broadsides of the Victory and the Temeraire...\nMachine translation: Alle unsere eigenen Waffen wurden entweder zerschlagen oder von den Breitseiten des Sieges und des Temeraire demontiert...\nReference translation: Alle unsere eigenen Waffen wurden entweder zerschlagen oder von den Breitseiten der Victory und der Temeraire vernichtet ... \nScore: 87"

        elif self.lang_pair_short == "et-en":
            examples = "Example 1\nSource: 74.\t“Ma ilmutasin end kotka kujul teadmiste puu kohal.\nMachine translation: 74. I have shown myself above the tree of knowledge in the form of a dog.\nReference translation: 74 . \ t \" I have revealed myself above the tree of knowledge in the form of an eagle . \nScore: 4\n\nExample 2\nSource: Siiski on allutatud rusikareeglil \"muretse viletsuse kõrvaldamise, mitte õnne edendamise pärast\" häid külgi.\nMachine translation: However, the rule of the rusty is subject to 'concerns about the elimination of destitution, not the promotion of happiness'.\nReference translation: However , the subordinate rule of thumb is to \" worry about eliminating misery , not promoting happiness \" .\nScore: 33\n\nExample 3\nSource: Nii et võime kohust täita sõltuks kalduvustes, mis loodus juhtus meile andma.\nMachine translation: So we can fulfil the obligation by the leanings that nature has given us.\nReference translation: So the ability to fulfil an obligation would depend on the tendencies that nature happened to give us .\nScore: 49\n\nExample 4\nSource: Lend sümboliseeris fašistliku juhtimise jõudu ning tööstuslikku ja tehnoloogilist edu, mida riik fašistliku juhtimise all saavutas.\nMachine translation: It was symbolised by the power of fascist leadership and by the industrial and technological progress achieved by the country under fascist control.\nReference translation: The flight symbolised the power of fascist leadership and the industrial and technological progress achieved by the country under fascist control .\nScore: 63\n\nExample 5\nSource: Sihtmärgiks valiti Viljandi, ülestõusu peamine keskus.\nMachine translation: Viljand, the main centre of uprising, was chosen as a target.\nReference translation: Viljandi , the main centre of uprising , was chosen as a target .\nScore: 86"

        elif self.lang_pair_short == "ne-en":
            examples = "Example 1\nSource: मगरहरुले मनाउने विशेष चाड मध्ये माघेसक्रांति यस भेगका मगरहरूले पनि विशेष रुपमा मनाउने गर्दछन् ।\nMachine translation: The Maggots of this vegetation typically consist of a special feast that is celebrated by the maggots.\nReference translation: Among the special festivals celebrated by the Magars , Maghe Sankranti is also celebrated in a special way by the Magars of this region .\nScore: 17\n\nExample 2\nSource: त्यसवेला कम्युनिस्ट पार्टीको कर्णाली प्रदेश इन्चार्जमा शैलेन्द्रकुमार उपाध्याय थिए भने सेक्रेटरीमा पाल्पाका कमलराज रेग्मी ।\nMachine translation: At that time, the communist party's karnali province was enchanted with shaikumar, the lesser regime of palpa in the secular.\nReference translation: At that time , Shailendra Kumar Upadhyaya was the in-charge of Karnali Province and Kamalraj Regmi was the Secretary .\nScore: 30\n\nExample 3\nSource: सन् १२९१मा तीनवटा मुलुकबीच सम्झौता भई महासङ्घ बनेको थियो।\nMachine translation: In 1291, an agreement was made between the three roads to the historians.\nReference translation: In 1291 , an agreement was reached between the three countries to form a federation .\nScore: 58\n\nExample 4\nSource: यस विषयको खोज अनुसन्धान गर्दा यो कुरा थाहा भएको हो ।\nMachine translation: It is known when investigating the findings of this topic.\nReference translation: This was known when it was investigated on this subject .\nScore: 74\n\nExample 5\nSource: यसपछि डोटीको अधिनता स्वीकार गरेर बझाङ्ग राज्यले बस्नु पर्यो ।\nMachine translation: Soon afterwards, the bazhang kingdom had to abide by accepting the subordination of doti.\nReference translation: Afterwards , the Bajhang Kingdom had to abide by accepting the subordination of Doti .\nScore: 82"

        elif self.lang_pair_short == "ro-en":
            examples = "Example 1\nSource: a activat în domeniul glam rockului în același deceniu, ei constituind sursa de inspirație a formației americane de hard rock Guns N' Roses.\nMachine translation: KAD and KAD were jointly responsible for the violent repression of civil society following the December 2010 elections and for the repression of civil society following the December 2010 elections.\nReference translation: he worked in the field of glam rock in the same decade , being the source of inspiration for the American hard rock band Guns N ' Roses .\nScore: 3\n\nExample 2\nSource: Divizia de infanterie ușoară cu sediul al Vught era singura forță de manevră a armatei olandeze.\nMachine translation: Fire only force in the Dutch army to manoeuvre the easy infantry division with its headquarters.\nReference translation: The Vught-based light infantry division was the only maneuvering force of the Dutch army .\nScore: 38\n\nExample 3\nSource: Pe 10 mai 1940, britanicii au ocupat Islanda și Insulele Faroe.\nMachine translation: On 10 May 1940, the British dealt with Iceland and the Islands.\nReference translation: On May 10 , 1940 , the British occupied Iceland and the Faroe Islands .\nScore: 51\n\nExample 4\nSource: La început, cel mai folosit tip de coif era modelul Montefortino, care a fost folosit cel puțin din secolul III î.H., dacă nu IV î.H.\nMachine translation: At the beginning, the most used type of coif was the Montefortino model, which was used from at least the third century H., if not IV-H..\nReference translation: In the beginning , the most used type of helmet was the Montefortino model , which was used at least from the 3rd century BC , if not IV BC .\nScore: 66\n\nExample 5\nSource: Deoarece unitățile de dischetă se instalează de obicei în compartimente de aceeași înălțime\nMachine translation: As disk units are usually installed in compartments of the same height\nReference translation: Because floppy disk drives are usually installed in compartments of the same height\nScore: 94"

        elif self.lang_pair_short == "ru-en":
            examples = "Example 1\nSource: Бодливой корове бог рог не дает.\nMachine translation: God does not give a cow a horn.\nReference translation: God does not give the horn to the butting cow .\nScore: 19\n\nExample 2\nSource: Пристало, как седло к корове.\nMachine translation: Like a saddle to a cow.\nReference translation: Like a saddle to a cow .\nScore: 21\n\nExample 3\nSource: Пытался склеить кнопки клеем но в итоге нихуя не склеил и разлил клей на 4 блятских кнопки и теперь они в клею\nMachine translation: I tried to glue the buttons but in the end I didn't glue them together and poured glue on 4 blat buttons and now they are in glue\nReference translation: I tried to glue the buttons but in the end I didn 't glue them on right and poured glue on 4 fucking buttons and now they are covered in glue\nScore: 43\n\nExample 4\nSource: Проплаченная массовка и бюджетники под предводительством Гундяева собрались на митинг за строительство храма святой Екатерины у Театра Драмы в Екатеринбурге.\nMachine translation: The paid masses and public servants, led by Gundyaev, gathered for a rally for the construction of St. Catherine's Church near the Drama Theatre in Yekaterinburg.\nReference translation: The paid masses and public servants , led by Gundyaev , gathered for a rally for the construction of St. Catherine 's Church near the Drama Theatre in Yekaterinburg .\nScore: 78\n\nExample 5\nSource: Разбор решения ЕСПЧ по Навальному :: \nMachine translation: Review of the ECHR's decision on Navalny:\nReference translation: Analysis of the ECHR decision on Navalny : : \nScore: 85"

        elif self.lang_pair_short == "si-en":
            examples = "Example 1\nSource: බොහෝ නූතන වගකීම් රක්ෂණ මෙසේ ඒ වෙනුවට ගෙවීමේ ව්යදවහාරයෙන් ලියා තිබේ.\nMachine translation: Most modern liability insurance table was written in the constitution of payment instead.\nReference translation: Most modern liability insurance is written in the practice of payment instead .\nScore: 18\n\nExample 2\nSource: ඒ අයට තමයි විශේෂ බළකා සෙබළෙකුට තිබිය යුතු පුහුණුවීම් ලබලා දෙන්නේ.\nMachine translation: Those are the practices that a special cavalry person should have.\nReference translation: They are the ones who provide the training that a Special Forces soldier should have .\nScore: 23\n\nExample 3\nSource: පිරිමින් දසදහස් ගණනින් ජපන් හමුදාවට බලෙන් බඳවා ගැනිණි.\nMachine translation: Men were forced to recruit the Japanese Army in tens of thousands.\nReference translation: Tens of thousands of men were forcibly recruited into the Japanese army .\nScore: 55\n\nExample 4\nSource: වසර 80 තුළ මෙම ජාතීන් අතර සම්බන්ධතා ශක්තිමත් වූ අතර කොරියාවේ ඉන්පසු රජ වූ සියලු රජවරුන් විවාහ වූයේ මොංගෝලියානු කුමරියන් සමගයි.\nMachine translation: In the 80 years, relations between these nations were strong and all the kings of Korea were married to the mongolian princesses.\nReference translation: During the 80s , relations between these nations were strong , and all subsequent kings of Korea married Mongolian princesses .\nScore: 73\n\nExample 5\nSource: විශේෂයෙන් පරාක්ෂග්ධන භාණ්ඩ භාවිතාවක් තුළින් නිෂ්පාදනයේ ගුණත්වය සහ අලංකාරත්වයෙන් යුතුව වෙළඳපොලට මුදා හැරීමෙන් සිය වෙළඳපොල බලය තර කර ගැනීමට උත්සහ කරයි.\nMachine translation: Especially the quality of the product in use of a range of products and elements released into the market will try to compete with hundreds of market power.\nReference translation: Especially Seeks to strengthen its market power by launching products with quality and elegance , especially through the use of ? ? ? ? products .\nScore: 83"

        else:
            raise ValueError("This language pair is not supported yet!")
        
        for i in range(len(self.main_file)):
            instruction = "You are going to evaluate the quality of machine translation given the source, machine translation and reference translation. The followings are examples of scoring translation quality. \n\n%s\n\nNow score the following translation from %s to %s with respect to the human reference and examples above on a continuous scale from 0 to 100, where score of zero means \"no meaning preserved\" and score of one hundred means \"perfect meaning and grammar\".\nSource: %s\nMachine translation: %s\nReference translation: %s\nScore:" % (
                examples,
                source,
                target,
                self.main_file["src"][i],
                self.main_file["mt"][i],
                self.main_file["ref"][i],
                )
            instruction = self.format.format(user_input=instruction)
            instruction_list.append(instruction)

        df = pd.DataFrame({"final_prompt": instruction_list})
        df.to_csv(self.lang_pair_short + "_vllm_t08.tsv", index=True, encoding="utf-8", sep="\t")


def main():

    args = parse_args()

    languauge_pairs = ["en-de", "en-mr", "en-zh", "et-en", "ne-en", "ro-en", "ru-en", "si-en"]

    for lang_pair in languauge_pairs:
        if args.main_file:
            main_file = args.main_file
        else:
            main_file = "raw_data/" + lang_pair + "_overlaps_test.tsv"
        previous_output_file = None
        
        template = VllmTemplate(main_file, error_file=None, previous_output_file=previous_output_file, format=args.prompt_format)
        template.generate_template01()
        template.generate_template02()
        template.generate_template03()
        template.generate_template04()
        template.generate_template05()
        template.generate_template06()
        template.generate_CoT_prompt1()
        template.generate_template08()

        #template.generate_CoT_prompt2()

if __name__ == "__main__":
    main()
