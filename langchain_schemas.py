# PROPERTIES
SENTIMENT_PROPERTY = "conflict_sentiment"
CEASEFIRE = "ceasefire"
HUMANITARIAN_PAUSE = "humanitarian pause"
SETTLER_VIOLENCE = "Settler violence"
IRAN = "Iran"
SUPPLIES = "Supplies"
AMERICAN_FINANCIAL_AID = "American financial aid"
HOSTAGES = "Hostages"
OCTOBER_7 = "October 7"
PALESTINIAN_CASUALTIES = "Palestinian casualties"
STANCE_TOWARD_PRESIDENT_BIDEN_ADMINISTRATION_POLICY = "Stance toward President Biden Administration Policy"
THE_DAY_AFTER_THE_WAR = "the day after the war"

RELEASE_ALL_HOSTAGES = "release all hostages"
DEFEAT_HAMAS = "defeat Hamas"
HOSTAGES_RELEASE_AGREEMENT = "hostages release agreement"

#PROPERTIES PER TASK
ISRAEL_SENTIMENT_TASK_PROPERTIES = [SENTIMENT_PROPERTY]

MULTI_LABEL_TASK_PROPERTIES = [
    RELEASE_ALL_HOSTAGES
]

HOUTHIS_SENTIMENT  = "houthis_sentiment"
HOUTHIS_TASK_PROPERTIES = [HOUTHIS_SENTIMENT]



# DESCRIPTIONS
# US
basic_classification_desc = '''
In recent weeks, a conflict has been ongoing between Israel and Hamas in the Gaza Strip. This conflict began following a Hamas atrocity attack on Israel, resulting in the tragic loss of more than 1,300 civilian lives and soldiers. In response, Israel initiated airstrikes and ground operation in Gaza. Public opinion regarding this war is evolving daily. Your task is to classify tweets related to the Gaza-Israel war into one of the following categories:
  Class 1, Class 2, Class 3, Class 0, or Class 9. Below, we provide clear and detailed guidance for each class.
  Class 1: Condemn Israeli attacks, refer to Israeli attacks as war crimes, or demand a ceasefire.
  - Tweets in this category should be identified when they explicitly criticize Israel and Israeli military actions or refer to them as war crimes.
  - Also, tweets that call for a ceasefire without necessarily condemning Israel should be classified as Class 1. The focus is on Gaza humanitarian crisis and ending the conflict fast as possible.ֹ
- Tweets that opposing giving aid to Israel because Israel use that aid against Palestinians. Pay attention, tweets that opposing giving aid to Israel because of the controversy between Republicans and Democrats about IRS funding and taxes should not be label as class 1.
- Pay attention, many tweets will discuss the need for humanitarian aid to Gaza and humanitarian pause. It is not mean that are immediately belong to this class. Many pro-Israeli tweets also see the need for keep Palestinians civilians out of the war circle. On the other hand if the tweet emphasize things like war crimes that Israel does it may suggest that it should be label as class 1.
  Class 2: Expressing a complex attitude towards the relationship between Israel and Palestine.
  - This class is for tweets that don't fall neatly into Class 1 or Class 3 but involve nuanced opinions regarding the conflict.
  - Look for tweets discussing the need for humanitarian aid or humanitarian pause in Gaza without explicitly condemning Israel. These tweets express complex attitudes by recognizing the humanitarian aspect of the situation.
- the tweets differentiate between Hamas as a terrorist organization and innocent people in gaza who needs humanitarian help.
- tweets that represent a wish to give humanitarian aid to gaza, maybe by egypt.
  Class 3: Support Israel or condemn Hamas as terrorists.
  - Tweets that express support for Israel or label Hamas as a terrorist organization should be classified under Class 3.
- Tweets that condemn Hamas terrorist attack against Israel, including the murder of civilians, families and babies and the capturing of hostages.
- tweets that refer to Hamas as ISIS or as evil jihadists.
  - Ensure that the tweet demonstrates a clear stance of support for Israel and or strong condemnation of Hamas as terrorists.
  - Emphasize Hamas responsibility to the situation in Gaza.
  Class 0: Unrelated to the conflict.
  - If a tweet is unrelated to the Gaza-Israel conflict, it should be classified as Class 0.
  - Tweets discussing unrelated topics or not mentioning the conflict in any way should be placed in this category.
- pay attention: Tweets that refer to internal events  in the united states, as example the murder of 6 years old Palestinian-American girl should be in class 0
-	Class 9: Relating to the conflict between Israel and Gaza, but focus on internal American issues and politics without taking a clear side in the conflict between Israel and Gaza.    If a tweet is related to Gaza-Israel conflict, but it has ambiguous massage which cannot be inferred, it should be classified as Class 9
-	If a tweet is related to Israel aid, but it is focusing on the American domestic political controversy between Republicans and Democrats about taxes and IRS funding it should be labeled as class 9. Pay attention that not every controversy between Democrats and Republicans should be labeled as class 9. If the controversy is not connected to Israel aid it should be labeled as class 0.
-	Tweets that their main focus is condemning Islamophobia and Antisemitism inside the United States (for example in universities) and not focus about the support for Israel or Gaza should be label as class 9
  Examples:
  Tweet: "Russia &amp;Hamas are both killers.Stand with #Biden and  #freedom and #democracy . This is clearly the right policy for #USA #Ukraine RussiaWar #IsrealPalestineconflict"
  ANSWER: 3
  Tweet: "I stand with the Palestinian people. My heart aches for the families who have lost loved ones. We can not stand as silent witnesses to children"
  ANSWER: 1
  Tweet: "I am closely monitoring the situation in Israel with great concern.  My heartfelt condolences go out to the families who have tragically lost their loved ones in this abhorrent attack on innocent civilians.  This is absolutely unacceptable."
  ANSWER: 3
 Tweet: "Hamas made terrorist attacks against Israel.
The need for humanitarian assistance in Gaza."
  ANSWER: 2
Tweet: "The NDAA passed with more support from Democrats than Republicans after their extreme attempts to play political games with our national security."
    ANSWER: 0
Tweet: "Handing $1T to for-profit war contractors who have failed their audits 6 years in a row while gutting safety net programs families rely on is cruel policy. Today, I voted NO on FY24 Defense Authorization, a budget that promotes never-ending war at the expense of working families."
    ANSWER: 0
Tweet: {input}
    ANSWER:    
'''

# EU
basic_classification_desc_europe = '''
In recent weeks, a conflict has been ongoing between Israel and Hamas in the Gaza Strip. This conflict began following a Hamas atrocity attack on Israel, resulting in the tragic loss of more than 1,400 civilian lives and soldiers. In response, Israel initiated airstrikes and ground operation in Gaza. Public opinion regarding this war is evolving daily. Your task is to classify tweets related to the Gaza-Israel war into one of the following categories: 
    Class 1, Class 2, Class 3, Class 0, or Class 9. Below, we provide clear and detailed guidance for each class.
    Class 1: Condemn Israeli attacks, refer to Israeli attacks as war crimes, or demand a ceasefire.
        - Tweets in this category should be identified when they explicitly criticize Israel and Israeli military actions or refer to them as war crimes.
        - Also, tweets that call for a ceasefire without necessarily condemning Israel should be classified as Class 1. The focus is on Gaza humanitarian crisis and ending the conflict fast as possible.ֹ
        - Pay attention, many tweets will discuss the need for humanitarian aid to Gaza and humanitarian pause. It is not mean that are immediately belong to this class. Many pro-Israeli tweets also see the need for keep Palestinians civilians out of the war circle. On the other hand if the tweet emphasize things like war crimes that Israel does it may suggest that it should be label as class 1.
    Class 2: Expressing a complex attitude towards the relationship between Israel and Palestine.
        - This class is for tweets that don't fall neatly into Class 1 or Class 3 but involve nuanced opinions regarding the conflict.
        - Look for tweets discussing the need for humanitarian aid or humanitarian pause in Gaza without explicitly condemning Israel. These tweets express complex attitudes by recognizing the humanitarian aspect of the situation.
        - the tweets differentiate between Hamas as a terrorist organization and innocent people in gaza who needs humanitarian help.
        - tweets that represent a wish to give humanitarian aid to gaza, maybe by egypt.
    Class 3: Support Israel or condemn Hamas as terrorists.
        - Tweets that express support for Israel or label Hamas as a terrorist organization should be classified under Class 3. 
        - Tweets that condemn Hamas terrorist attack against Israel, including the murder of civilians, families and babies and the capturing of hostages. 
        - tweets that refer to Hamas as ISIS or as evil jihadists.
        - Ensure that the tweet demonstrates a clear stance of support for Israel and or strong condemnation of Hamas as terrorists.
        - Emphasize Hamas responsibility to the situation in Gaza.
    Class 0: Unrelated to the conflict.
        - If a tweet is unrelated to the Gaza-Israel conflict, it should be classified as Class 0.
        - Tweets discussing unrelated topics or not mentioning the conflict in any way should be placed in this category.
    Class 9: Relating to the conflict between Israel and Gaza, but it has ambiguous massage which cannot be inferred or focus on internal USA issues
        - If a tweet is referring to the conflict but it has ambiguous massage which cannot be inferred, it should be classified as Class 9.
        - Tweets that their main focus is condemning Islamophobia and Antisemitism (for example in protests and demonstrations) and not focus about the support for Israel or Gaza should be label as class 9
    
    Examples:
    Tweet: "Russia &amp;Hamas are both killers.Stand with #Biden and  #freedom and #democracy . This is clearly the right policy for #USA #Ukraine RussiaWar #IsrealPalestineconflict"
    ANSWER: 3
    
    Tweet: "I stand with the Palestinian people. My heart aches for the families who have lost loved ones. We can not stand as silent witnesses to children"
    ANSWER: 1
    
    Tweet: "I am closely monitoring the situation in Israel with great concern.  My heartfelt condolences go out to the families who have tragically lost their loved ones in this abhorrent attack on innocent civilians.  This is absolutely unacceptable."
    ANSWER: 3
    
    Tweet: "Hamas made terrorist attacks against Israel.
    The need for humanitarian assistance in Gaza."
    ANSWER: 2
    
    Tweet: {}
    Answer:
'''


houthis_classification_desc = '''
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

# SCHEMAS
ISRAEL_SENTIMENT_TASK_SCHEME = {
    "properties": {
        SENTIMENT_PROPERTY: {
            "type": "integer",
            "enum": [0,1,2,3,9],
            "description": basic_classification_desc,
        },
    },
    "required": ISRAEL_SENTIMENT_TASK_PROPERTIES,
}

ISRAEL_SENTIMENT_TASK_SCHEME_EUROPE = {
    "properties": {
        SENTIMENT_PROPERTY: {
            "type": "integer",
            "enum": [0,1,2,3,9],
            "description": basic_classification_desc_europe,
        },
    },
    "required": ISRAEL_SENTIMENT_TASK_PROPERTIES,
}


MULTI_LABEL_TASK_SCHEME = {
    "properties": {
        RELEASE_ALL_HOSTAGES: {
            "type": "integer",
            "enum": [0, 1],
            "description": "Calling for the complete release of all hostages held by Hamas, and not just part of them. It may include #BringThemALLHome or #BringThemHome hashtags. Pay attention, discussing the release of hostages should not be automatically labelled in this class if it doesn't include clear call for the complete release of all the hostaged held by Hamas.",
        },
    },
    "required": MULTI_LABEL_TASK_PROPERTIES,
}

HOUTHIS_TASK_SCHEME = {
    "properties": {
        HOUTHIS_SENTIMENT: {
            "type": "integer",
            "enum": [1, 2,3],
            "description": houthis_classification_desc
        },
    },
    "required": HOUTHIS_TASK_PROPERTIES,
}

# SIMILARITY
ISRAEL_SENTIMENT_TASK_SIMILARITY_QUERY_TITLE = "israel_related"
ISRAEL_SENTIMENT_TASK_SIMILARITY_QUERY = "The ongoing conflict between Israel and Gaza involves military actions, political tensions, and humanitarian concerns"

HOUTHIS_TASK_SIMILARITY_QUERY_TITLE = "houthis_related"
HOUTHIS_TASK_SIMILARITY_QUERY = \
    "Houthi attacks on commercial ships in the Red Sea. The Houthis are targeting Israeli-owned ships. " \
     "In response to this, the US and the UK carry out airstrikes on Houthi targets in Yemen."

HOUTHIS_TASK_KEYWORDS = ['Houthis', 'Houthi', 'Red Sea', 'Proxies', 'Rebels', 'Yemen', 'Yemenis', 'Vessel', 'Proxy', 'Rebel']
# UNRWA

UNRWA_SENTIMENT  = "unrwa_sentiment"
UNRWA_TASK_PROPERTIES = [UNRWA_SENTIMENT]

unrwa_prompt = '''

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
    
'''

UNRWA_TASK_SCHEME = {
    "properties": {
        UNRWA_SENTIMENT: {
            "type": "integer",
            "enum": [0,1,2,3],
            "description": unrwa_prompt
        },
    },
    "required": UNRWA_TASK_PROPERTIES,
}


UNRWA_TASK_SIMILARITY_QUERY_TITLE = "unrwa_related"
UNRWA_TASK_SIMILARITY_QUERY = """allegations claimimg that UNRWA employees have participated in October 7 events in israel 
and gaza have led to the halt of U.S. funding of the organization.
"""