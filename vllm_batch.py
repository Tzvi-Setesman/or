from pathlib import Path
import pandas as pd
from vllm import LLM, SamplingParams

# regex filters per Task (default to no regex filter)
filter_dict = {
    'unrwa' : 'UNRWA|unrwa'
}

task = 'houthis_filter' # houthis # 'unrwa' 
FILTER_DF_SIM_EMB = False # Filter rows that have similarity_class_col == 1 --> into LLM
filter_regx = filter_dict.get(task, None)
orig_label_col = f'{task}_sentiment'
similarity_class_col = f'{task}_related_class_from_similarity'

model_name_or_path = 'NousResearch/Hermes-2-Pro-Llama-3-8B' # 'mistralai/Mistral-7B-Instruct-v0.2' # 'openchat/openchat-3.5-1210'
DEBUG_FILTER_ALL_ZEROS = False # False - all data for inference. True - filter data (remove 0 class) before inference 

# Read texts to classify
label_col_name = f'{task}_sentiment'
input_folder_path = f'data/{task}/input'
output_folder_path = f'data/{task}/classified_out'
print(f'**** Started run: task={task}, model_name_or_path={model_name_or_path}')
file_name_no_ext = '2024-02-17_19-25-06__2024-02-22_15-12-37_HouthisTweet_similarity'
file_ext = '.xlsx'
# input_file_path is the full path to input file or empty / None to loop all files in <input_folder_path>
input_file_path =  None # f'{input_folder_path}/{file_name_no_ext}{file_ext}' 

exp_suffix = '' # suffix to add to .parquet to describe exp hyper params (model, temperature, prompt fixes ...)
debug_suffix = '_no_0' # add to file name suffix to describe what debug mode is different 

unrwa_prompt = """
In recent times, allegations have surfaced regarding cooperation between some UNRWA employees and Hamas during rounds of fighting in Gaza. Consequently, American aid to UNRWA has been stopped.
Your task is to classify tweets from members of the American Congress regarding the cessation of aid to the United Nations Relief and Works Agency for Palestine Refugees in the Near East (UNRWA) into one of the following categories:
Each time a tweet is provided, I will respond with the corresponding category number it belongs to.
Class 0: Tweets not related to UNRWA funding.
- Tweets mentioning UNRWA or Gaza without discussing the halt of UNRWA funding
-Tweets that talk about hunger or a difficult humanitarian situation and do not specifically mention UNRWA
-Tweets that talk about a certain budget or funding but do not talk about the organization UNRA

Class 1: Tweets expressing opposition to UNRWA and advocating for ending funding.
- Tweets condemning UNRWA and supporting the cessation of aid, citing concerns over alleged connections with Hamas.
- Statements welcoming the halt in aid as necessary to prevent support for alleged terrorist activities.

Class 2: Tweets mentioning the decision to halt funding for UNRWA, without expressing a specific opinion.
- Tweets discussing the cessation of aid to UNRWA without firmly taking a stance either in favor or against it.
- Comments acknowledging the cessation of aid to UNRWA but not providing a clear opinion on whether it should be supported or opposed.

Class 3: Tweets in favor of UNRWA and advocating for continued funding.
- Tweets supporting UNRWA and opposing the cessation of aid, emphasizing the importance of its humanitarian efforts.
- Statements advocating for the restoration of funding to UNRWA to ensure continued support for Palestinians.

Examples of tweets and the classification that suits them:

"Trump halted ALL funding to the UNRWA. 
Biden immediately restored funding and has paid over a BILLION DOLLARS to the Hamas supporters since taking office. 
It’s estimated that nearly half of all UNRWA employees have ties to Hamas and other terrorist groups. 
UNACCEPTABLE.…"
ANSWER 1:

"The United States should not have given one dollar to the antisemitic @UNRWA. 
According to a new report, approximately 1,200 staff members have ties to Hamas or other terrorist organizations."
ANSWER 1:

"I joined Sen. @MarshaBlackburn in introducing legislation after October 7 to block American taxpayer dollars going to the UNRWA, because the U.S. has a national security imperative to ensure that not a single cent of our money enables Hamas.
What took the Biden Admin this long?"
ANSWER 1:

"As the largest humanitarian organisation in the #GazaStrip - we will do whatever possible to continue our indispensable work to support the people of 📍#Gaza 
At this critical moment #DonateToUNRWA: https://t.co/BAncZ2TnRb https://t.co/tY6VQrdsHM"
ANSWER 3:

"Cutting off support to @UNRWA - the primary source of humanitarian aid to 2 million+ Gazans - is unacceptable.
Among an organization of 13,000 UN aid workers, risking the starvation of millions over grave allegations of 12 is indefensible.
The US should restore aid immediately."
ANSWER 3:

"Yesterday, I met with the leaders of several major humanitarian NGOs who just returned from travel to Gaza. I remain concerned about the humanitarian crisis in Gaza, and I'll keep working for lifesaving aid for Gazan civilians. https://t.co/krJHGXh5ne"
ANSWER 0: 

"Sitting Israeli officials spoke at a conference to recolonize Gaza, calling for the expulsion of Palestinians to build new Israeli settlements. The U.S. cannot continue to be complicit in ethnic cleansing of Palestinians. President Biden must stop funding it."
ANSWER 0:

"Gaza is starving. I spoke with @WFPChiefEcon Arif Husain and the bottom line is this: If we do not see a dramatic improvement in humanitarian access very soon, countless more innocent people – including thousands of children – could die. The U.S. must act. https://t.co/mIqKmAE8Hd"
ANSWER 0:


"Great point from @RepMikeRogersAL.
We need a FULL accounting of exactly where US taxpayer dollars are going after they are spent by the UN AND whether any US dollars were WRONGFULLY used to perpetrate the terrorist attack on Oct. 7th."
ANSWER 0:
    
"{input}"
ANSWER

"""
houthis_prompt = '''
In recent weeks, Yemeni Houthi militants, which are backed by the iranian regime, have repeatedly targeted and attacked U.S. personnel, military bases, and commercial vessels in the Red Sea, resulting in significant disruptions to maritime traffic and raising concerns about regional stability and security. As a response, the U.S., in collaboration with several other countries, has initiated Operation Prosperity Guardian to collectively address security challenges in the southern Red Sea and the Gulf of Aden, with the primary goal of ensuring the freedom of navigation in the area. On January 12, 2024, U.S. and U.K. forces conducted multiple airstrikes against Houthi targets as part of this operation.

According to Lt. Gen. Douglas Sims, the director of the US military's Joint Staff, the initial strikes on January 12 successfully achieved their objective of damaging the Houthis' ability to launch complex drone and missile attacks similar to the ones they conducted earlier. However, despite causing significant damage to Houthi capabilities, subsequent assessments indicated that the strikes did not entirely deter further attacks on shipping in the region. Yemeni sources reported casualties among Houthi fighters, including members of Hezbollah, the Iranian Revolutionary Guards, and Iraqi militants.

Reactions to these military actions have been varied. Domestically, the Yemeni government strongly condemned the Houthi attacks and affirmed its right to enhance security in the Red Sea region. On the other hand, Houthi officials labeled the airstrikes as blatant aggression and vowed retaliation. Massive protests erupted in Houthi-controlled cities denouncing the U.S. and British military actions.

Internationally, responses ranged from expressions of support to condemnation. While some countries, including the U.S. and the U.K., justified the strikes as necessary for safeguarding maritime navigation and addressing security threats, others, such as Iran and Syria, condemned them as violations of Yemen's sovereignty and international law. Notably, China urged restraint and highlighted the need to avoid further escalation in the region.

Given these developments and the diverse range of reactions, your task is to classify tweets related to American strikes against the Houthis into one of the following categories: Class 1, Class 2, Class 3, Class 4, Class 5, or Class 0. Below, we provide clear and detailed guidance for each class based on the spectrum of opinions and reactions observed.

Class 1: Supportive of U.S. actions against the Houthis.

Tweets that endorse U.S. and coalition operations against the Houthis, emphasizing the need for repercussions for Houthi aggression.
Statements welcoming military actions as necessary and overdue, highlighting the Biden administration's warnings to the Houthis.
Expressions of support for attacking the Houthis in response to their actions, linking it to national security concerns.
Pay attention, tweets with minor criticism of the Biden administration, as long as they primarily endorse the military actions against the Houthis, should still be classified under Class 1.
If the tweet explicitly endorses targeting Tehran as a response to Iranian support for militant groups or aggression against U.S. interests or allies, it should be classified as Class 1.
Class 2: Expressing a nuanced view on the American attacks against the Houthis.

Tweets presenting mixed opinions on U.S. attacks in Yemen on the Houthis, acknowledging both positive and negative aspects without firmly aligning with either support or objection.
Pay attention, if the tweet primarily expresses support for the strike and emphasizes the importance of taking action against the Houthi aggression, while also containing elements of critique towards the Biden administration's handling of the situation in a manner that does not include constitutional concerns, regional stability concerns, or isolationist views, should be classified under Class 1, as long as the overall tone of the tweet leans more towards endorsing the military actions.
Class 3: Objecting to military operations against the Houthis due to potential escalation of violence.

Tweets expressing concerns about the consequences of military operations against the Houthis, including escalation of violence and destabilization of the region.
Statements criticizing airstrikes against the Houthis as violating international law or sovereignty, with a focus on promoting peace and stability.
Tweets expressing concerns about the effectiveness of military actions in Yemen against the Houthis and advocating for alternative, non-violent approaches to conflict resolution.
Class 4: Opposing the attacks against the Houthis because of opposing any U.S. involvement in foreign conflicts, reflecting isolationist views.

Tweets condemning U.S. attacks against the Houthis because of objection to U.S. involvement in foreign wars and advocating for prioritizing domestic affairs over military engagements.
You will be classified in this category only if the tweet talks about it in the context of the Houthis or the attacks in Yemen. If it is talking about other places, do not classify as this category.
Class 5: Opposing the American attacks on the Houthis, citing constitutional concerns.

Tweets explicitly condemning U.S. airstrikes against the Houthis as unconstitutional due to lack of congressional approval.
Statements asserting that only Congress has the authority to declare war and criticizing the Biden administration for bypassing this process in the Houthi subject.
Expressions of reluctance to support broader military involvement in Yemen without clear congressional authorization and debate.

Class 0: Unopinionated tweets or tweets that are not related to the american houthi conflict
Tweets that do not express any opinion on the houthi american situation and the american attacks on the houthi rebeles.
Include tweets that are not related to the houthies and their actiones.
expressions that are related to different conflicts apart from the houthies will also be included in class 0.
Pay attention, tweets that mention the need to release all hostages and the war in Gaza without mentioning Houthi attacks in the Red Sea should be classified under class 0.
Pay attention, if the tweet contains the words "Houthis," "Red Sea," "Iranian proxies," and such, it should be classified under class 0 if it does not express an opinion regarding the Houthi attacks or the American attacks. If a tweet contains those words without expressing an opinion, it should be classified under class 0.
- Tweets that only focus on the south U.S border immigrants issue - class 0
- Tweets that only focus on U.S economy or internal issues - class 0
- Very short tweets (1-3 words) - class 0


Examples:
Tweet: "This action by U.S. and British forces is long overdue, and we must hope these
		operations indicate a true shift in the Biden Administration’s approach to Iran and its
		proxies that are engaging in such evil and wreaking such havoc. They must
		understand there is a serious price to pay for their global acts of terror and their
		attacks on U.S. personnel and commercial vessels. America must always project
		strength, especially in these dangerous times."
ANSWER: 1

Tweet: "I welcome the U.S. and coalition operations against the Iran-backed Houthi terrorists
		responsible for violently disrupting international commerce in the Red Sea and
		attacking American vessels. President Biden’s decision to use military force against
		these Iranian proxies is overdue."
ANSWER: 1

Tweet: "The United States and our allies must leave no room to doubt that the days of unanswered terrorist 
		aggression are over."
ANSWER: 1

Tweet: "The United States carries a special, historic obligation to help protect and defend
		these arteries of global trade and commerce. And this action falls directly in line with
		that tradition. That is clearly reflected in both our national security strategy and the
		national defense strategy. It is a key conviction of the president and it is a
		commitment that we are prepared to uphold."
ANSWER: 1

Tweet: "The administration’s desperate attempts at non-escalation have invited escalation. Their appeasement approach to the terrorists...has given them a green light to attack us, attack international shipping, and expand the war further."
ANSWER: 1

Tweet: I was glad to see President Biden finally place the Houthis back on the terror list after removing them at the beginning of his term. We should go a step further by finally passing and signing the SHIP Act into law. https://t.co/M1VQ9rHdP9
ANSWER: 1

Tweet:
The Houthis have always been a terrorist organization — the Biden administration should never have delisted them in the first place.

This change needs to be effective immediately.
ANSWER: 1

Tweet: "The President’s strikes in Yemen are unconstitutional. For over a month, he
		consulted an international coalition to plan them, but never came to Congress to
		seek authorization as required by Article I of the Constitution. We need to listen to
		our Gulf allies, pursue de-escalation, and avoid getting into another Middle East war."
ANSWER: 5

Tweet: "The United States cannot risk getting entangled into another decades-long conflict
		without Congressional authorization. The White House must work with Congress
		before continuing these airstrikes in Yemen."
ANSWER: 5

Tweet: "We will see if these strikes deter Iran and its proxies from further attacks; I have my
		doubts. History teaches that only devastating retaliation will deter Iran, as when
		President Trump killed their terrorist mastermind in 2020 and President Reagan sank
		half their navy in 1988. That bold, decisive action is the opposite of what we’ve seen
		from Joe Biden for three years."
ANSWER: 2

Tweet: "In this particular instance, I do support military reprisals and military attacks to deter
		attacking our ships, but [the Biden administration] shouldn’t be allowed to do that
		without permission."
ANSWER: 5

Tweet: "the strikes showed a &quot;complete disregard for international law&quot; and were
		&quot;escalating the situation in the region."
ANSWER: 3

Tweet: "The U.S. air strikes on Yemen are another example of the Anglo-Saxons&#39; perversion
		of UN Security Council resolutions."
ANSWER: 3

Tweet: "These attacks are a clear violation of Yemen&#39;s sovereignty and territorial integrity,
		and a breach of international laws. These attacks will only contribute to insecurity
		and instability in the region."
ANSWER: 3

Tweet: "I Called for restraint and &quot;avoiding escalation&quot; after the strikes and said it was
		monitoring the situation with &quot;great concern&quot;."
ANSWER: 3

Tweet: "I call on all parties involved not to escalate even more the situation in the interest of
		peace and stability in the Red Sea and the wider region."
ANSWER: 3

Tweet: "U.S. Strikes Won’t Achieve What a Gaza Cease-Fire Could in the Middle East"
ANSWER: 3

Tweet: "This is why I called for a ceasefire early. This is why I voted against war in Iraq,  and why I am against the attacks in Yemen today.
		Violence only begets more violence."
ANSWER: 3

Tweet: "We need a ceasefire now to prevent deadly, costly, catastrophic escalation of
		violence in Yemen.
ANSWER: 3

Tweet: "I would not support us being pulled into a broader war because of the operation in Yemen."
ANSWER: 3

Tweet: "I have what some may consider a dumb idea, but here it is:  stop the bombing of Gaza, then the attacks on commercial shipping will end. Why not try that approach?"
ANSWER: 3

Tweet: "As representatives of the American people, Congress must engage in robust debate
		before American servicemembers are put in harm’s way in Yemen and before more U.S.
	   taxpayer dollars are spent on yet another war in the Middle East."
ANSWER: 4

Tweet: "Do not spent dollars in the MiddleEast before you secured internal America affairs and economy"	   
ANSWER: 4

Tweet: "Only Congress has the power to declare war. I have to give credit to [Rep. Ro
		Khanna] here for sticking to his principles, as very few are willing to make this
		statement while their party is in the White House."
ANSWER: 5

Tweet: "The President must come to Congress for permission before going to war.
		Biden can not solely decide to bomb Yemen. 
		And what is the condition of Secretary of Defense Lloyd Austin? Is he still laid up in
		the hospital?
		Biden [administration] wants to fund war in Ukraine, control the war in Israel, arm
		Taiwan and prep for war with China, and is now going to war in the Middle East???
		All with a wide open border, millions invading, and millions of got aways?! 
		This is insanely out of control!"
ANSWER: 5

Tweet: "his is where we should put party aside and stand for the oath we all took: Congress
		alone decides if we go to war. I join my colleagues on both sides insisting we follow
		the Constitution."
ANSWER: 5

Tweet: "POTUS is violating Article I of the Constitution by carrying out airstrikes in Yemen
		without congressional approval. The American people are tired of endless war."
ANSWER: 5

Tweet: "The President must come to Congress for permission before going to war. Biden can not
		solely decide to bomb Yemen."
ANSWER: 5

Tweet: "Biden is  weak. This country needs a change.  vote for Trump 2024."
ANSWER: 0

Tweet: Ramadan Mubarak to all observing this sacred month.

This Ramadan is particularly difficult, w millions of Palestinians on the brink of starvation and under siege. We must secure a ceasefire to save countless lives.

May the blessings of this month bring a just and lasting peace".
	 ANSWER: 0


Tweet: 
-I urge @POTUS and partners in the region to continue with negotiations to reach a mutual ceasefire agreement that is in the best interests of Israel, Palestinians, and stability and peace in the region. The current situation in the Middle East is not sustainable.
ANSWER: 0

Tweet: 
- Our nation is facing over $34 trillion of debt, and our southern border is wide open. Yet the D.C. Cartel continues to prioritize foreign nations while putting America last. We must secure America’s borders first!

Mayorkas was the first ever Homeland Security Secretary to be impeached by the House."
ANSWER: 0


Tweet: 
-"Without a vote from Congress, the Biden Administration is sending another $300 million to fund the war in Ukraine.

This is on top of the $113 BILLION in funding American taxpayers have already sent over there. 

America should be worried about securing our OWN border, not Ukraine’s. 
https://t.co/x1PzCoVjd8"
ANSWER: 0

Tweet:
Rather than supporting small businesses with pro-growth policies, @POTUS imposed unnecessary red tape on "joint employers," a move that will kill jobs and stifle economic growth.
Proud to vote in support of @RepJames's resolution to nullify @POTUS's attack on small businesses.
ANSWER: 0

Tweet:
The fact of the matter is we’re dealing with an issue much bigger than our budget.

We have an invasion underway at our border and if it continues to go unaddressed, it won’t matter what our topline spending number is or how much we cut because we won’t have a country left.
ANSWER: 0

Tweet:
President Biden’s weakness has only emboldened Iran and its proxies to continue launching missiles and attacking Americans and our allies. 

Indecisive half-measures won’t stop them. Only devastating retaliation will. https://t.co/IL8bRr5dzW
ANSWER: 1

Tweet:
Three years ago, Joe Biden made the weak decision to remove the Houthi's terrorist designation -- another failed attempt at appeasing Iran-backed forces who chant 'Death to America.' With hostile adversaries testing American resolve, U.S. strength is essential for our national security.

ANSWER: 1

Tweet: us
ANSWER: 0

Tweet: {input}
Answer:
'''

houthis_filter_prompt = """
You are a machine that categorizes tweets from the US Congress. Classified into the following 2 categories:

Category 0: Not Related to American Attacks on Houthis
Classify the tweet as Category 0 if it does not mention or discuss the American airstrikes against Houthi targets in Yemen. This includes tweets about unrelated topics or events.
Category 1: Related to American Attacks on Houthis
Classify the tweet as Category 1 if it discusses, references, or provides information about the American and allied airstrikes targeting Houthi positions in Yemen. This includes tweets that mention the attacks, the targets, the military operations, or the ongoing conflict between the US/allies and the Houthis.
The key factors to look for are explicit mentions of the American attacks, the Houthis as the target, and the location of Yemen.

Here are some examples:
Tweet: Any Hamas apologists who wish to take advantage of the Iran-backed Houthi Terrorism Scholarship Program should book a one-way flight to Yemen.
ANSWER: 0
Tweet: #Israel is facing the threat of imminent attack, directly from Iran and in combination with coordinated attacks by Hezbollah, Houthis & Iranian proxies in Syria & Iraq 
Biden must stop the harsh criticism of the only pro-American democracy in the region & make clear the U.S. will support them in defending against and responding to any such attacks
ANSWER: 0
Tweet: I just voted for legislation that will re-designate the Houthis as a Foreign Terrorist Organization (FTO) and ensure proper sanctions remain on Iran. If Biden won’t respond swiftly to the continued aggression of Iran and its proxies, the @HouseGOP will.
ANSWER: 0
Tweet: I support payback military operations against the Houthis 
ANSWER: 1

Classify the following: format as ANSWER : <0 or 1> only
Tweet: {input}
ANSWER:
"""

prompt = globals()[f'{task}_prompt']  # Note the Tweet: {input} place holder was deleted and instead, formatted below


##### End Config ##### 
def parse_response(resp: str):
    import re
    lst_str_nums = re.findall(r'(?:ANSWER:\s*)?(\b\d+\b)', resp)
    numbers = [int(str_num) for str_num in lst_str_nums]
    return numbers
        
def get_class(resp):
    if not type(resp) == str:
        return 0 # return 0 for those didn't pass similarity filter
    numbers = parse_response(resp)
    num = numbers[0] if len(numbers) > 0 else -1
    return num
        
def batch_inference(prompt, input_file_path, llm):
    print(f'*** batch_inference {input_file_path}')
    
    output_file_path = f'{output_folder_path}/{Path(input_file_path).stem}{exp_suffix}' 
    if DEBUG_FILTER_ALL_ZEROS:
        output_file_path+=debug_suffix
    output_file_path += file_ext

    # If input_file_path already has a valid output_file_path file - skip it     
    p_out = Path(output_file_path)
    if p_out.exists() and p_out.stat().st_size > 0:
        print(f'*** Skip {input_file_path}, output file {p_out} already exist')
        return 
        
    df = pd.read_excel(input_file_path)       
    if orig_label_col in df.columns:
        df = df.rename(columns={orig_label_col: f'gpt4_{orig_label_col}'})

    if DEBUG_FILTER_ALL_ZEROS:
        df = df[(df.gpt4_houthis_sentiment != '0') & (df.gpt4_houthis_sentiment != 'Azure content filter')]

    # Filter not related Tweets -  by filter classifer (embs) as similarity=0    
    df['row_included'] = True
    if FILTER_DF_SIM_EMB:
        df['row_included'] = df['row_included'] & df[similarity_class_col] == 1
        
        
    # Filter by regexp (kmusa)    
    if filter_regx:
        df['included_regex'] = df.post_text.str.contains(filter_regx,regex=True)
        df['row_included'] = df.row_included & df.included_regex
    
    # Combine both filters to a row_included True/False col - if True - prompt 
    df_filtered = df[df.row_included]

    # Post processing of regex filter to df already classified with prompts    
    #df.loc[~df.row_included,'unrwa_sentiment'] = 0    
    #df.to_excel(r"D:\NLP\Netivot\experiments\data\unrwa\classified_out\2024-02-05_17-14-22__2024-04-01_18-39-12_Unrwa_classified.xlsx")
        
    generating_prompts = df_filtered.post_text.apply(lambda text: prompt.replace('{input}',text)).tolist()

    print("-" * 80)

    # The llm.generate call will batch all prompts and send the batch at once
    # if resources allow. The prefix will only be cached after the first batch
    # is processed, so we need to call generate once to calculate the prefix
    # and cache it.
    outputs = llm.generate(generating_prompts[0], sampling_params)

    # Subsequent batches can leverage the cached prefix
    outputs = llm.generate(generating_prompts, sampling_params)

    # Gather outputs 
    gen_texts = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        gen_texts.append(generated_text)
        
    # Assign generated text (from LLM response text), only to the rows with similarity_class_col == 1 
    # those were sent to LLM    
    df['gen_resp'] = pd.NA        
    df.loc[df.row_included, 'gen_resp'] = gen_texts
    
    # Parse gen text --> class (int)
    df[label_col_name] = df.gen_resp.apply(get_class)
    
    # Save results     
    df.to_excel(output_file_path)

if __name__ == "__main__":
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=10)

    # Create an LLM.
    llm = LLM(model=model_name_or_path)

    if not input_file_path is None and len(input_file_path) > 10:
        batch_inference(prompt, input_file_path, llm)
    else:
        # Loop all .xlsx files in <input_folder_path>
        lst_files = list(Path(input_folder_path).glob(f'*{file_ext}'))
        if len(lst_files) == 0:
            print(f'*** Warning: Did not find files with *.{file_ext} under {input_folder_path}')
        for full_path in lst_files:
            batch_inference(prompt, full_path, llm)
    