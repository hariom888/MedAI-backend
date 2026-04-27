"""
Master Rebuild Script
Parses all diseases from the embedded text data and regenerates:
  - disease_templates.json
  - rag_disease_db.json
  - symptom_vocab.json
  - label_encoder.json
  - feature_dictionary.json
  - class_distribution.json
  - train.csv / test.csv
  - All rag_chunks/*.md files
"""

import re, json, csv, os, random
from collections import defaultdict

from excluded_diseases import is_excluded_disease

random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# DISEASE TEXT — full embedded dataset from the document
# ─────────────────────────────────────────────────────────────────────────────

DISEASE_TEXT = r"""
Disease: Alzheimer's disease
Aliases: alzheimer disease, AD, senile dementia of alzheimer type
Description: progressive neurodegenerative disorder and most common cause of dementia, leading to decline in memory, thinking, behavior, and ability to perform daily activities
Cause: accumulation of amyloid-beta plaques and tau protein tangles in brain, along with age-related neuronal degeneration and genetic, lifestyle, and environmental factors
Transmission: none
Risk Groups: adults over 65, individuals with family history, people with traumatic brain injury, individuals with hearing loss, depression, or mild cognitive impairment
Incubation Period: none
Symptoms:
- memory loss, especially recent events, repeating questions, forgetting names and conversations
- confusion, disorientation, getting lost in familiar places
- difficulty speaking, reading, writing, planning, and decision-making
- behavioral changes such as anxiety, aggression, low mood, hallucinations, delusions
- difficulty performing daily tasks, poor self-care, reduced mobility in later stages
- progressive cognitive decline leading to complete dependence in advanced stages
Progression:
1. mild memory loss and subtle cognitive decline
2. increasing confusion, language difficulties, and impaired daily functioning
3. severe memory loss, personality changes, and need for assistance
4. complete dependence, loss of communication, and extensive brain function decline
Common Locations: brain regions responsible for memory, thinking, and language (hippocampus, cerebral cortex)
Duration: chronic lifelong condition with gradual progression over years
Severity: severe
Complications: severe cognitive impairment, inability to perform daily activities, infections, malnutrition, dehydration, wandering, injury, caregiver stress
Home Remedy: cognitive stimulation, regular physical activity, healthy diet, social engagement, structured routine, supportive care environment
Avoid: smoking, social isolation, unmanaged chronic conditions, head injuries, poor sleep habits
When to See a Doctor: noticeable memory loss, confusion, behavioral changes, difficulty performing routine tasks, concerns about cognitive decline
Emergency Signs: severe confusion, inability to recognize close family, sudden behavioral changes, dehydration, injury from wandering, inability to eat or drink
Prevention: regular exercise, healthy diet, mental stimulation, social engagement, managing chronic diseases, avoiding smoking, adequate sleep
Contagious Period: none
Special Notes: not a normal part of aging, early-onset cases occur before age 65 but are rare, disease progresses gradually and requires long-term care, mild cognitive impairment may precede condition
===
Disease: Parkinson's disease
Aliases: parkinson disease, PD, idiopathic parkinsonism
Description: progressive neurodegenerative movement disorder caused by degeneration of dopamine-producing neurons, leading to impaired motor control and non-motor symptoms
Cause: loss of nerve cells in substantia nigra resulting in dopamine deficiency, associated with alpha-synuclein protein aggregation (lewy bodies), influenced by genetic and environmental factors
Transmission: none
Risk Groups: adults over 50, males, individuals with family history, people exposed to toxins or pesticides, individuals with prior head injury
Incubation Period: none
Symptoms:
- tremor at rest, often starting in hands, fingers, foot, or jaw
- bradykinesia (slowed movement), difficulty initiating movement and performing daily tasks
- muscle rigidity, stiffness, reduced arm swing, and painful muscle tension
- balance problems, poor posture, increased risk of falls
- speech changes, soft or slurred speech, monotone voice
- loss of automatic movements such as blinking or smiling
- writing changes, small and cramped handwriting
- non-motor symptoms including depression, anxiety, sleep disturbances, constipation, fatigue, memory problems, loss of smell
Progression:
1. mild tremor and subtle motor symptoms, often on one side of body
2. increasing rigidity, slowed movement, and balance issues
3. worsening motor impairment and development of non-motor symptoms
4. severe disability, difficulty walking, speaking, and performing daily activities
Common Locations: brain regions controlling movement, especially substantia nigra and basal ganglia
Duration: chronic lifelong condition with gradual progression over years to decades
Severity: moderate_to_severe
Complications: dementia, depression, swallowing difficulties, malnutrition, sleep disorders, falls, infections, orthostatic hypotension, bladder problems
Home Remedy: regular exercise, physiotherapy, balanced diet, adequate sleep, stress management, maintaining daily routine
Avoid: exposure to toxins, physical inactivity, poor diet, unmanaged stress, sleep deprivation
When to See a Doctor: presence of tremors, stiffness, slowed movement, balance issues, or noticeable changes in speech or coordination
Emergency Signs: frequent falls, severe swallowing difficulty, inability to move, confusion, sudden worsening of symptoms
Prevention: regular aerobic exercise, healthy diet, avoiding toxins, possible protective effects of caffeine and certain medications
Contagious Period: none
Special Notes: symptoms often begin asymmetrically and worsen over time, no cure available but medications and surgery can manage symptoms, life expectancy is usually near normal with treatment
===
Disease: Migraine
Aliases: migraine headache, vascular headache, migraine disorder
Description: recurrent neurological disorder characterized by moderate to severe throbbing headaches, often one-sided, associated with sensory sensitivity and systemic symptoms
Cause: abnormal brain activity affecting nerve signals, neurotransmitters, and blood vessels, with strong genetic predisposition and multiple environmental and lifestyle triggers
Transmission: none
Risk Groups: women, individuals with family history, people with depression, anxiety, epilepsy, sleep disorders, hormonal fluctuations, and obesity
Incubation Period: none
Symptoms:
- throbbing or pulsing headache, usually on one side of the head, sometimes bilateral
- nausea, vomiting, sensitivity to light, sound, and odors
- aura symptoms such as flashing lights, zig-zag lines, numbness, tingling, or speech difficulty
- confusion, dizziness, poor concentration, fatigue, and weakness
- worsening pain with movement, coughing, or physical activity
- post-episode exhaustion, confusion, and weakness lasting hours to a day
Progression:
1. prodrome phase with mood changes, cravings, yawning, or fluid retention
2. aura phase with visual or sensory disturbances
3. headache phase with increasing throbbing pain and associated symptoms
4. postdrome phase with fatigue, confusion, and residual discomfort
Common Locations: head (often unilateral), may affect face and neck regions
Duration: 4 hours to 72 hours per episode, with possible residual symptoms up to several days
Severity: moderate_to_severe
Complications: medication overuse headache, chronic migraine, stroke risk, depression, anxiety, sleep disturbances
Home Remedy: rest in dark quiet room, hydration, cold compress, stress management, regular sleep schedule, trigger avoidance
Avoid: stress, sleep deprivation, skipped meals, excessive caffeine, alcohol, strong lights or smells, trigger foods, medication overuse
When to See a Doctor: frequent or severe headaches, worsening pattern, unclear diagnosis, headaches interfering with daily activities
Emergency Signs: sudden severe worst headache, paralysis or weakness, slurred speech, seizures, high fever with stiff neck, confusion, vision loss
Prevention: regular exercise, balanced diet, adequate sleep, stress management, avoiding triggers, maintaining routine, preventive medications if frequent attacks
Contagious Period: none
Special Notes: may occur with or without aura, more common in women due to hormonal influence, attacks can be predictable in some individuals, keeping a migraine diary helps identify triggers
===
Disease: Aphasia
Aliases: post-stroke aphasia, language disorder due to brain injury, acquired language impairment
Description: neurological communication disorder caused by brain damage (commonly stroke) affecting language production and comprehension while intelligence remains intact
Cause: damage to language-dominant brain regions (usually left hemisphere, including broca's and wernicke's areas) due to stroke or other brain injury
Transmission: none
Risk Groups: stroke patients, elderly individuals, people with brain injury, individuals with neurological disorders affecting left brain hemisphere
Incubation Period: none
Symptoms:
- difficulty speaking, forming sentences, or producing correct words
- trouble finding words (anomia), fragmented or jumbled speech
- impaired understanding of spoken language
- difficulty reading, writing, or performing calculations
- problems with long or complex words and sentences
- frustration, confusion, and possible depression due to communication difficulties
Progression:
1. sudden onset after stroke or brain injury
2. severe communication impairment in early stage
3. gradual improvement with therapy in some cases
4. persistent language deficits depending on severity and brain damage
Common Locations: left hemisphere of brain, especially broca's area and wernicke's area
Duration: varies from temporary (weeks to months) to permanent depending on severity of brain damage
Severity: moderate_to_severe
Complications: depression, social isolation, communication barriers, reduced quality of life, difficulty in daily functioning
Home Remedy: speech therapy exercises, communication practice, supportive environment, use of visual aids and assistive communication tools
Avoid: social isolation, lack of therapy, stress, communication neglect
When to See a Doctor: sudden difficulty speaking, understanding language, or reading/writing, especially after suspected stroke
Emergency Signs: sudden speech loss, confusion, weakness on one side, facial drooping, inability to understand speech (possible stroke signs)
Prevention: stroke prevention through blood pressure control, healthy diet, exercise, avoiding smoking, managing diabetes and cholesterol
Contagious Period: none
Special Notes: does not affect intelligence, includes types such as global aphasia, broca's aphasia, and wernicke's aphasia, may coexist with apraxia, early rehabilitation improves outcomes
===
Disease: Multiple sclerosis
Aliases: MS, disseminated sclerosis
Description: chronic autoimmune neurological disorder affecting the central nervous system, causing demyelination and disruption of nerve signal transmission between brain and body
Cause: autoimmune attack on myelin sheath of nerve fibers, influenced by genetic susceptibility and environmental factors
Transmission: none
Risk Groups: women, adults aged 20-40, individuals with family history, people with autoimmune conditions
Incubation Period: none
Symptoms:
- visual disturbances such as blurred vision or double vision
- muscle weakness and reduced strength
- coordination and balance problems, difficulty walking
- numbness, tingling, or pins and needles sensations
- fatigue and reduced physical endurance
- cognitive issues including memory and thinking difficulties
Progression:
1. early mild neurological symptoms and sensory disturbances
2. relapsing-remitting episodes with partial recovery
3. gradual worsening with increased disability in some cases
4. advanced stage with significant motor and cognitive impairment
Common Locations: brain, spinal cord, optic nerves
Duration: lifelong condition with variable progression
Severity: mild_to_severe
Complications: mobility impairment, vision loss, cognitive decline, speech difficulties, muscle stiffness, disability
Home Remedy: regular exercise, balanced diet, stress management, adequate rest, physical and occupational therapy
Avoid: stress, overheating, smoking, physical inactivity
When to See a Doctor: vision problems, persistent numbness, coordination issues, unexplained weakness, or neurological symptoms
Emergency Signs: sudden vision loss, severe weakness, loss of coordination, inability to walk, severe neurological deterioration
Prevention: none
Contagious Period: none
Special Notes: often follows relapsing-remitting pattern, more common in women, no cure but treatments can slow progression and manage symptoms, early diagnosis improves outcomes
===
Disease: Meningitis
Aliases: meningeal infection, brain membrane inflammation
Description: infection or inflammation of the meninges (protective membranes surrounding brain and spinal cord), causing rapid onset neurological and systemic symptoms
Cause: viral (most common), bacterial, fungal, or parasitic infections, and in some cases autoimmune conditions or medications
Transmission: respiratory droplets, saliva, nasal secretions, fecal contact, close contact such as kissing or sharing utensils
Risk Groups: infants, children, elderly, immunocompromised individuals, unvaccinated people, people in close living conditions, travelers to endemic regions
Incubation Period: typically 2-10 days for bacterial and viral forms
Symptoms:
- sudden fever, severe headache, nausea, vomiting
- neck stiffness, sensitivity to light (photophobia), double vision
- confusion, altered mental state, difficulty concentrating
- seizures, drowsiness, difficulty waking
- rash in some bacterial types such as meningococcal
- infants: bulging fontanelle, irritability, poor feeding, lethargy
Progression:
1. initial flu-like symptoms (fever, malaise)
2. rapid onset of neurological symptoms (headache, stiff neck, confusion)
3. severe complications such as seizures, shock, or organ failure
4. recovery or progression to permanent neurological damage or death if untreated
Common Locations: meninges surrounding brain and spinal cord
Duration: acute condition, typically days to weeks depending on type and treatment
Severity: moderate_to_severe
Complications: brain damage, hearing loss, vision loss, seizures, stroke, kidney failure, shock, cognitive impairment, death
Home Remedy: none
Avoid: close contact with infected individuals, sharing utensils or personal items, poor hygiene
When to See a Doctor: sudden fever, severe headache, neck stiffness, confusion, sensitivity to light, vomiting, or suspected exposure
Emergency Signs: seizures, unconsciousness, severe confusion, high fever with stiff neck, rash with fever, difficulty breathing, shock
Prevention: vaccination (pneumococcal, meningococcal, haemophilus influenzae type b), good hygiene, avoiding close contact, prophylactic antibiotics for exposed individuals
Contagious Period: varies by cause, bacterial forms contagious during active infection until treated
Special Notes: bacterial meningitis is life-threatening and requires immediate antibiotics, viral meningitis is usually less severe, early treatment significantly improves outcomes
===
Disease: Encephalitis
Aliases: brain inflammation, viral encephalitis, autoimmune encephalitis
Description: inflammation of brain tissue caused by infections or immune-mediated processes, leading to neurological dysfunction and potential brain damage
Cause: viral infections (e.g., herpes simplex, arboviruses, enteroviruses), bacterial, fungal, or parasitic infections, or autoimmune response where immune system attacks brain cells
Transmission: varies by cause, includes respiratory droplets, saliva, contaminated surfaces, insect bites (mosquitoes, ticks), or none in autoimmune cases
Risk Groups: all age groups, immunocompromised individuals, people with HIV, individuals exposed to infected insects, travelers to endemic regions
Incubation Period: varies depending on cause, typically 4-14 days for viral forms
Symptoms:
- fever, headache, fatigue, body aches
- confusion, altered thinking, memory loss, impaired judgment
- seizures, tremors, myoclonic jerks, or abnormal movements
- speech, hearing, or vision problems including double vision or loss
- neck stiffness, sensitivity changes, partial paralysis
- behavioral changes, hallucinations, personality changes
- severe cases: loss of consciousness, coma
Progression:
1. initial mild flu-like symptoms or asymptomatic phase
2. onset of neurological symptoms such as confusion and seizures
3. worsening brain inflammation leading to severe dysfunction
4. recovery with possible residual deficits or progression to brain damage or death
Common Locations: brain (cerebral tissue), may involve spinal cord in some cases
Duration: acute phase lasts 1-2 weeks, recovery may take weeks to months
Severity: moderate_to_severe
Complications: brain damage, memory loss, seizures, paralysis, speech and vision loss, cognitive impairment, coma, death
Home Remedy: none
Avoid: exposure to infected individuals, insect bites, poor hygiene, contact with contaminated objects
When to See a Doctor: fever with neurological symptoms such as confusion, seizures, speech difficulty, or behavioral changes
Emergency Signs: seizures, loss of consciousness, paralysis, severe confusion, breathing difficulty, sudden vision or speech loss
Prevention: vaccination (measles, mumps, rubella, rabies, varicella), mosquito and tick protection, hygiene, avoiding contact with infected individuals
Contagious Period: varies by infectious cause, contagious during active infection in some viral types
Special Notes: can be infectious or autoimmune, rapid progression requires immediate medical care, early treatment improves survival, rehabilitation may be needed for recovery
===
Disease: Central Nervous System Tumor
Aliases: brain tumor, spinal cord tumor, CNS tumor, brain cancer
Description: abnormal growth of cells in the brain or spinal cord that may be benign or malignant, affecting neurological function and body control systems
Cause: uncontrolled cell growth in CNS tissues, often unknown, associated with genetic mutations, inherited syndromes, environmental exposures, or metastasis from other cancers
Transmission: none
Risk Groups: adults, individuals with genetic syndromes (NF1, NF2, Li-Fraumeni, tuberous sclerosis), people exposed to toxins, individuals with weakened immune systems, cancer patients with metastasis risk
Incubation Period: none
Symptoms:
- persistent or morning headaches, often relieved after vomiting
- seizures and abnormal neurological activity
- vision, hearing, or speech difficulties
- nausea, vomiting, and loss of appetite
- personality, mood, or behavioral changes
- memory, concentration, or cognitive impairment
- balance problems, difficulty walking, weakness or numbness in limbs
- spinal symptoms: back pain, bowel or bladder dysfunction
Progression:
1. initial localized tumor growth with mild or no symptoms
2. increasing pressure on brain or spinal cord causing neurological symptoms
3. tumor expansion or spread within CNS affecting multiple functions
4. severe neurological impairment, disability, or life-threatening complications
Common Locations: brain (cerebrum, cerebellum, brainstem), spinal cord, meninges
Duration: chronic condition with variable progression depending on tumor type and grade
Severity: moderate_to_severe
Complications: brain damage, paralysis, seizures, cognitive decline, vision or hearing loss, recurrence, metastasis, death
Home Remedy: none
Avoid: exposure to harmful chemicals, radiation, unmanaged genetic risks
When to See a Doctor: persistent headaches, seizures, neurological deficits, vision or speech problems, unexplained behavioral changes
Emergency Signs: seizures, loss of consciousness, sudden paralysis, severe headache, vision loss, inability to speak or move
Prevention: none (risk reduction includes managing genetic conditions and minimizing toxin exposure)
Contagious Period: none
Special Notes: includes many tumor types (gliomas, meningiomas, ependymomas), may be primary or metastatic, grading (I-IV) determines aggressiveness, treatment includes surgery, radiation, chemotherapy, and targeted therapy
===
Disease: Bell's palsy
Aliases: acute peripheral facial palsy, idiopathic facial nerve paralysis
Description: sudden onset neurological condition causing temporary weakness or paralysis of facial muscles on one side due to facial nerve dysfunction
Cause: inflammation and swelling of the facial nerve (cranial nerve VII), often linked to viral infections such as herpes simplex, varicella-zoster, or other respiratory viruses
Transmission: none
Risk Groups: pregnant women, individuals with diabetes, hypertension, obesity, people with recent viral or respiratory infections, family history of recurrence
Incubation Period: none
Symptoms:
- sudden weakness or paralysis on one side of the face developing within hours to days
- facial drooping, difficulty smiling, closing eye, or making expressions
- drooling and difficulty controlling saliva
- pain around jaw or behind ear on affected side
- increased sensitivity to sound (hyperacusis)
- headache and loss of taste
- changes in tear and saliva production
Progression:
1. sudden onset of facial weakness or paralysis
2. peak severity within 48-72 hours
3. gradual improvement over weeks
4. recovery within months, though some may have residual symptoms
Common Locations: facial nerve (cranial nerve VII), affecting one side of face
Duration: weeks to months, typically resolves within 3-6 months
Severity: mild_to_moderate
Complications: permanent facial weakness, synkinesis (involuntary muscle movements), eye dryness, corneal damage, partial or complete vision loss in severe cases
Home Remedy: facial exercises, eye protection (lubricating drops, eye patch), rest, maintaining facial muscle activity
Avoid: eye exposure without protection, delayed treatment, stress, untreated infections
When to See a Doctor: sudden facial weakness or drooping, difficulty speaking or closing eye, to rule out stroke
Emergency Signs: sudden paralysis with limb weakness, speech difficulty, confusion (possible stroke), inability to close eye leading to injury
Prevention: none
Contagious Period: none
Special Notes: not caused by stroke but may mimic stroke symptoms, most cases recover fully, recurrence is rare but possible, early treatment improves recovery outcomes
===
Disease: Diabetic neuropathy
Aliases: diabetic nerve damage, diabetic peripheral neuropathy, diabetic autonomic neuropathy
Description: complication of diabetes causing progressive nerve damage due to prolonged high blood sugar levels, affecting sensory, motor, and autonomic functions
Cause: chronic hyperglycemia and elevated blood lipids (triglycerides) leading to nerve fiber damage and impaired blood supply to nerves
Transmission: none
Risk Groups: individuals with diabetes (type 1 or type 2), long-standing uncontrolled blood sugar, obese individuals, people with high triglycerides
Incubation Period: none
Symptoms:
- numbness, tingling, burning, or pain in feet and legs, sometimes hands and arms
- reduced sensation leading to unnoticed injuries or ulcers
- muscle weakness and loss of coordination
- digestive problems such as nausea, constipation, or diarrhea
- bladder dysfunction and urinary issues
- sexual dysfunction including erectile problems
- abnormal sweating and temperature regulation
- dizziness due to blood pressure changes
Progression:
1. early nerve damage with mild tingling or numbness
2. increasing sensory loss and pain in extremities
3. spread to autonomic or focal nerve dysfunction
4. severe nerve damage causing disability and organ dysfunction
Common Locations: peripheral nerves (feet, legs, hands), autonomic nerves (heart, digestive system, bladder), focal nerves (single nerve areas)
Duration: chronic and progressive condition
Severity: moderate_to_severe
Complications: foot ulcers, infections, amputations, gastroparesis, erectile dysfunction, bladder issues, cardiovascular instability, hypoglycemia unawareness
Home Remedy: strict blood sugar control, healthy diet, regular exercise, foot care, proper hydration
Avoid: poor glucose control, smoking, alcohol, sedentary lifestyle, untreated injuries
When to See a Doctor: numbness, pain, burning sensation, digestive or urinary issues, or unexplained weakness in diabetic patients
Emergency Signs: severe infections, non-healing wounds, sudden dizziness or fainting, loss of sensation leading to injury
Prevention: good blood sugar control, regular monitoring, healthy lifestyle, routine foot and nerve checkups
Contagious Period: none
Special Notes: most common complication of diabetes, peripheral neuropathy is most prevalent type, early management can slow progression and prevent severe complications
===
Disease: Sciatica
Aliases: sciatic nerve pain, radicular pain, nerve root pain
Description: condition characterized by pain radiating along the sciatic nerve from the lower back through the buttock and down the leg due to nerve irritation or compression
Cause: compression or irritation of spinal nerve roots, commonly from herniated disc, bone spurs, spinal degeneration, or inflammation
Transmission: none
Risk Groups: adults aged 20-50, individuals with obesity, sedentary lifestyle, diabetes, occupations involving heavy lifting or prolonged sitting
Incubation Period: none
Symptoms:
- lower back pain radiating to buttock, thigh, and leg (usually one side)
- sharp, burning, or electric shock-like pain along nerve pathway
- numbness, tingling, or pins and needles in leg or foot
- muscle weakness in leg or foot
- pain worsened by sitting, coughing, sneezing, or movement
- altered sensations such as hot, cold, or shooting pain
Progression:
1. initial lower back discomfort or nerve irritation
2. radiating leg pain with sensory changes
3. possible muscle weakness and reduced mobility
4. gradual improvement in most cases, or chronic symptoms if untreated
Common Locations: lower back (lumbar spine), buttocks, back of thigh, calf, foot along sciatic nerve
Duration: typically resolves within 6-12 weeks, but may persist longer or recur
Severity: mild_to_moderate
Complications: chronic pain, nerve damage, muscle weakness, mobility issues, cauda equina syndrome
Home Remedy: gentle exercise, maintaining activity, posture correction, heat or cold therapy, gradual return to normal activities
Avoid: prolonged sitting, heavy lifting, poor posture, sudden movements, inactivity
When to See a Doctor: persistent pain beyond a few weeks, worsening symptoms, significant weakness, or interference with daily activities
Emergency Signs: loss of bowel or bladder control, severe leg weakness, numbness in groin area (cauda equina syndrome), sudden severe pain after trauma
Prevention: regular exercise, proper posture, weight management, safe lifting techniques, avoiding prolonged inactivity
Contagious Period: none
Special Notes: usually improves without surgery, often caused by disc-related changes, symptoms may fluctuate and recur, imaging not always required in early stages
===
Disease: Cerebral palsy
Aliases: CP, congenital motor disorder, non-progressive neurological disorder
Description: group of permanent neurological disorders affecting movement, posture, and muscle coordination due to abnormal brain development or early brain damage
Cause: brain damage or abnormal brain development occurring before birth, during birth, or early childhood, including oxygen deprivation, infections, genetic factors, or injury
Transmission: none
Risk Groups: premature infants, low birth weight babies, multiple births, infants with birth complications, maternal infections, children with early brain injury
Incubation Period: none
Symptoms:
- abnormal muscle tone (spasticity, rigidity, or floppiness)
- difficulty with movement, balance, and coordination
- delayed developmental milestones (sitting, crawling, walking)
- speech and communication difficulties
- swallowing and feeding problems
- vision impairments and sensory processing issues
- seizures (epilepsy) and learning disabilities
- fatigue, sleep disturbances, and behavioral problems
Progression:
1. early developmental delays in infancy
2. identification of abnormal muscle tone and movement patterns
3. persistent motor impairment with possible associated conditions
4. stable neurological condition with functional changes over time
Common Locations: brain (motor control regions), affecting muscles throughout body
Duration: lifelong condition (non-progressive but symptoms may change over time)
Severity: mild_to_severe
Complications: joint deformities, muscle contractures, dislocations, chronic pain, communication difficulties, feeding issues, epilepsy
Home Remedy: physical therapy, assistive devices, regular exercise, supportive care, speech and occupational therapy
Avoid: untreated complications, lack of therapy, physical inactivity, poor posture support
When to See a Doctor: delayed development, abnormal muscle tone, difficulty in movement, speech, or coordination in infants or children
Emergency Signs: seizures, severe feeding difficulty, breathing problems, sudden worsening of neurological symptoms
Prevention: prenatal care, vaccination during pregnancy, avoiding infections, safe childbirth practices, prevention of head injuries in infants
Contagious Period: none
Special Notes: not progressive but symptoms may evolve with age, early intervention improves outcomes, includes types such as spastic, dyskinetic, ataxic, and mixed cerebral palsy
===
Disease: Dementia
Aliases: cognitive decline syndrome, neurocognitive disorder
Description: group of symptoms caused by brain cell damage leading to decline in memory, thinking, behavior, and ability to perform daily activities
Cause: damage or loss of brain cells due to conditions such as alzheimer's disease, stroke, brain injury, or tumors
Transmission: none
Risk Groups: older adults, individuals with family history, people with brain injury, stroke, or neurological diseases
Incubation Period: none
Symptoms:
- memory loss, especially recent events, forgetting tasks or conversations
- difficulty performing familiar tasks such as dressing or cooking
- language problems, forgetting words or using incorrect words
- disorientation in time and place, getting lost easily
- poor judgment and difficulty with abstract thinking
- misplacing objects in unusual places
- personality and mood changes such as depression, irritability, or anxiety
- loss of initiative, reduced interest in activities
Progression:
1. mild cognitive impairment and memory lapses
2. increasing confusion and difficulty with daily tasks
3. significant behavioral and personality changes
4. severe cognitive decline with complete dependence
Common Locations: brain (areas responsible for memory, reasoning, and behavior)
Duration: chronic and progressive condition
Severity: moderate_to_severe
Complications: inability to perform daily activities, behavioral disturbances, depression, caregiver burden, increased risk of injury
Home Remedy: cognitive stimulation, structured routine, supportive environment, social engagement
Avoid: social isolation, unmanaged medical conditions, head injuries, stress
When to See a Doctor: memory loss affecting daily life, confusion, behavioral changes, difficulty performing routine tasks
Emergency Signs: sudden confusion, severe behavioral changes, inability to recognize people, possible stroke-like symptoms
Prevention: healthy lifestyle, mental stimulation, managing cardiovascular risk factors, avoiding head injuries
Contagious Period: none
Special Notes: not a normal part of aging, alzheimer's disease is most common cause, symptoms worsen over time, early diagnosis helps manage progression
===
Disease: Type 1 diabetes
Aliases: insulin-dependent diabetes, juvenile diabetes, autoimmune diabetes
Description: chronic autoimmune disease in which the pancreas produces little or no insulin, leading to high blood glucose levels and impaired energy utilization
Cause: autoimmune destruction of insulin-producing beta cells in the pancreas triggered by genetic and environmental factors
Transmission: none
Risk Groups: children and young adults, individuals with family history, people with genetic predisposition, individuals with certain environmental triggers
Incubation Period: none
Symptoms:
- frequent urination (polyuria)
- excessive thirst (polydipsia)
- increased hunger despite eating (polyphagia)
- unexplained weight loss
- fatigue and weakness
- blurred vision
- slow-healing sores and frequent infections
- symptoms of diabetic ketoacidosis such as nausea, vomiting, abdominal pain, fruity breath, and difficulty breathing
Progression:
1. autoimmune destruction of pancreatic beta cells
2. gradual insulin deficiency leading to rising blood glucose levels
3. onset of symptoms over days to weeks
4. lifelong insulin dependence with risk of acute and chronic complications
Common Locations: pancreas (beta cells), systemic effects on blood, nerves, kidneys, eyes, heart
Duration: lifelong condition
Severity: severe
Complications: diabetic ketoacidosis, hypoglycemia, nerve damage, kidney disease, cardiovascular disease, vision loss, infections
Home Remedy: blood glucose monitoring, healthy diet, regular physical activity, hydration, adherence to insulin therapy
Avoid: missed insulin doses, poor diet, sedentary lifestyle, unmanaged stress, infections
When to See a Doctor: symptoms of excessive thirst, urination, weight loss, fatigue, or suspected diabetes
Emergency Signs: diabetic ketoacidosis symptoms (vomiting, abdominal pain, rapid breathing, confusion), severe hypoglycemia, loss of consciousness
Prevention: none
Contagious Period: none
Special Notes: requires lifelong insulin therapy, symptoms often develop rapidly, careful management prevents complications, may be associated with other autoimmune diseases
===
Disease: Type 2 diabetes
Aliases: non-insulin-dependent diabetes, adult-onset diabetes, insulin resistance diabetes
Description: chronic metabolic disorder characterized by insulin resistance and relative insulin deficiency, leading to elevated blood glucose levels
Cause: combination of insulin resistance in body cells and reduced insulin production by the pancreas, influenced by genetic, lifestyle, and environmental factors
Transmission: none
Risk Groups: overweight or obese individuals, sedentary lifestyle, adults over 35, people with family history, certain ethnic groups, individuals with prediabetes or gestational diabetes
Incubation Period: none
Symptoms:
- increased thirst (polydipsia)
- frequent urination (polyuria), especially at night
- increased hunger (polyphagia)
- fatigue and weakness
- blurred vision
- slow-healing wounds or sores
- frequent infections (skin, urinary, fungal)
- numbness or tingling in hands and feet
- darkened skin patches (acanthosis nigricans)
Progression:
1. development of insulin resistance and mild blood sugar elevation (prediabetes)
2. gradual rise in blood glucose with few or no symptoms
3. onset of noticeable symptoms and diagnosis
4. long-term complications affecting organs if uncontrolled
Common Locations: pancreas, blood, and systemic effects on heart, kidneys, nerves, eyes, and blood vessels
Duration: lifelong condition
Severity: moderate_to_severe
Complications: cardiovascular disease, stroke, neuropathy, kidney disease, retinopathy, infections, amputations, cognitive decline
Home Remedy: healthy diet, weight management, regular physical activity, blood glucose monitoring
Avoid: sedentary lifestyle, high sugar intake, smoking, excessive alcohol, poor diet
When to See a Doctor: symptoms of high blood sugar, risk factors, or routine screening for early detection
Emergency Signs: severe hyperglycemia, confusion, dehydration, hypoglycemia (if on medication), loss of consciousness
Prevention: healthy weight, balanced diet, regular exercise, avoiding smoking, managing prediabetes
Contagious Period: none
Special Notes: often asymptomatic in early stages, lifestyle changes can delay or prevent onset, may require oral medications or insulin over time
===
Disease: Hypothyroidism
Aliases: underactive thyroid, thyroid hormone deficiency
Description: chronic endocrine disorder in which the thyroid gland fails to produce sufficient thyroid hormones, leading to a generalized slowing of metabolic processes affecting multiple organ systems
Cause: autoimmune destruction (most commonly Hashimoto's disease), thyroid inflammation, congenital absence or dysfunction of thyroid gland, surgical removal of thyroid, radiation therapy, certain medications, iodine imbalance, or disorders of pituitary or hypothalamus
Transmission: none
Risk Groups: women, individuals over 60 years, people with family history of thyroid disease, postpartum women, patients with autoimmune diseases, individuals with prior thyroid surgery or radiation exposure
Incubation Period: gradual onset over months to years
Symptoms:
- fatigue and lethargy
- unexplained weight gain
- intolerance to cold temperatures
- dry skin and thinning hair
- joint and muscle pain
- slow heart rate (bradycardia)
- depression or low mood
- heavy or irregular menstrual cycles
- infertility issues
- constipation
- slowed thinking or memory problems
Progression:
1. early mild hormone deficiency with subtle or no symptoms
2. gradual worsening of metabolic slowdown and symptom development
3. systemic involvement affecting multiple organs
4. severe untreated cases may lead to life-threatening complications such as myxedema coma
Common Locations: thyroid gland (primary), systemic effects on heart, brain, skin, muscles, and metabolism
Duration: lifelong (chronic condition requiring long-term management)
Severity: mild_to_severe
Complications: high cholesterol, cardiovascular disease, infertility, depression, peripheral neuropathy, goiter, and severe cases such as myxedema coma
Home Remedy: balanced diet, adequate iodine intake, regular exercise, stress management, adherence to medication
Avoid: iodine excess, skipping medication, untreated thyroid disorders, certain interfering medications without supervision
When to See a Doctor: persistent fatigue, unexplained weight gain, menstrual irregularities, depression, or suspected thyroid dysfunction
Emergency Signs: extreme drowsiness, confusion, hypothermia, slow breathing, or unconsciousness (possible myxedema coma)
Prevention: not always preventable; early screening in high-risk individuals, managing autoimmune conditions, and appropriate iodine intake
Contagious Period: none
Special Notes: diagnosis is confirmed via thyroid function blood tests (TSH, T3, T4); treatment typically involves lifelong hormone replacement therapy with levothyroxine
===
Disease: Hyperthyroidism
Aliases: overactive thyroid, thyrotoxicosis
Description: endocrine disorder in which the thyroid gland produces and releases excessive thyroid hormones, leading to an increased metabolic rate that affects multiple body systems
Cause: autoimmune stimulation of the thyroid (most commonly Graves' disease), overactive thyroid nodules, thyroid inflammation, excessive iodine intake, or rarely pituitary gland tumors
Transmission: none
Risk Groups: women, individuals under 40 or over 60, people with family history of thyroid disease, patients with autoimmune conditions, individuals consuming excess iodine
Incubation Period: gradual onset over weeks to months
Symptoms:
- unexplained weight loss despite normal or increased appetite
- rapid or irregular heartbeat (palpitations)
- nervousness, anxiety, irritability
- tremors (shaking hands or fingers)
- increased sweating and heat intolerance
- fatigue and muscle weakness
- difficulty sleeping (insomnia)
- changes in menstrual cycles (lighter or irregular periods)
- frequent bowel movements
- enlarged thyroid gland (goiter)
- eye symptoms (bulging eyes, dryness, double vision) in Graves' disease
Progression:
1. early mild increase in thyroid hormone levels with subtle symptoms
2. progressive metabolic acceleration affecting multiple systems
3. complications involving heart, bones, and nervous system
4. severe untreated cases may lead to thyrotoxic crisis (life-threatening emergency)
Common Locations: thyroid gland (primary), systemic effects on heart, eyes, bones, muscles, and metabolism
Duration: variable; may be temporary or chronic
Severity: mild_to_severe
Complications: atrial fibrillation, heart failure, osteoporosis, infertility, eye complications, and thyroid storm
Home Remedy: stress management, balanced diet, avoiding excess iodine, regular medical follow-up
Avoid: excessive iodine intake, untreated symptoms, self-medication without supervision
When to See a Doctor: persistent rapid heartbeat, unexplained weight loss, anxiety, tremors, or swelling in the neck
Emergency Signs: high fever, severe tachycardia, confusion, delirium (possible thyroid storm)
Prevention: not fully preventable; early screening in high-risk individuals and avoiding excess iodine intake
Contagious Period: none
Special Notes: diagnosis is confirmed with thyroid function blood tests (TSH, T3, T4) and imaging; treatment options include anti-thyroid medications, radioactive iodine therapy, beta blockers, or surgery
===
Disease: Goiter
Aliases: thyroid enlargement
Description: abnormal enlargement of the thyroid gland, which may occur with normal, increased, or decreased thyroid hormone production; often visible as swelling at the front of the neck
Cause: iodine deficiency (most common), autoimmune disorders such as Graves' disease and Hashimoto's disease, thyroid nodules, inflammation, certain medications, or genetic factors
Transmission: none
Risk Groups: women, people over 40 years, individuals with iodine deficiency, those with family history of thyroid disease, patients with autoimmune conditions
Incubation Period: gradual development over months to years
Symptoms:
- swelling at the front of the neck
- tightness in throat
- difficulty swallowing or breathing (large goiter)
- hoarseness
- cough
- symptoms of hypothyroidism or hyperthyroidism depending on cause
Progression:
1. initial mild thyroid enlargement
2. gradual increase in size
3. possible hormonal imbalance
4. compression of nearby structures in severe cases
Common Locations: thyroid gland (neck)
Duration: variable; may be temporary or chronic
Severity: mild_to_severe
Complications: airway obstruction, swallowing difficulty, thyroid dysfunction, nodules, rarely malignancy
Home Remedy: adequate iodine intake, balanced diet, regular monitoring
Avoid: iodine deficiency or excess, smoking, untreated thyroid conditions
When to See a Doctor: visible neck swelling, breathing or swallowing difficulty, or symptoms of thyroid imbalance
Emergency Signs: severe breathing difficulty or rapid swelling
Prevention: use of iodized salt, maintaining proper iodine intake
Contagious Period: none
Special Notes: diagnosis involves thyroid function tests and imaging; treatment depends on underlying cause and may include observation, medication, radioactive iodine, or surgery
===
Disease: Obesity
Aliases: overweight condition (severe), excessive body fat
Description: complex chronic condition characterized by excessive accumulation of body fat that increases the risk of multiple health problems, including cardiovascular, metabolic, and musculoskeletal disorders
Cause: imbalance between calorie intake and energy expenditure, influenced by genetic, metabolic, behavioral, and environmental factors
Transmission: none
Risk Groups: individuals with sedentary lifestyle, high-calorie diet, family history of obesity, people with hormonal disorders, individuals with poor sleep, high stress, or certain medication use
Incubation Period: gradual development over months to years
Symptoms:
- excessive body fat (BMI 30 or above)
- increased waist circumference
- fatigue
- reduced physical endurance
- shortness of breath with activity
- joint or back pain
- sweating more than usual
Progression:
1. overweight stage (BMI 25-29.9)
2. obesity stage (BMI 30 or above)
3. increasing fat accumulation and metabolic imbalance
4. development of associated diseases and complications
Common Locations: systemic (affects entire body, especially fat tissue, heart, liver, joints)
Duration: chronic (long-term condition)
Severity: mild_to_severe
Complications: heart disease and stroke, Type 2 diabetes, high blood pressure and cholesterol, sleep apnea, osteoarthritis, fatty liver disease, certain cancers, depression and reduced quality of life
Home Remedy: balanced low-calorie diet, regular physical activity, behavioral changes, adequate sleep, stress management
Avoid: high-calorie processed foods, sugary beverages, sedentary lifestyle, excessive screen time
When to See a Doctor: BMI 30 or above, difficulty losing weight, or presence of obesity-related health issues
Emergency Signs: severe breathing difficulty, chest pain, or complications like stroke or heart attack
Prevention: healthy diet, regular exercise, maintaining active lifestyle, monitoring weight
Contagious Period: none
Special Notes: diagnosis is commonly based on BMI and waist circumference; treatment may include lifestyle modification, medications, and in severe cases, bariatric surgery
===
Disease: Metabolic Syndrome
Aliases: insulin resistance syndrome, syndrome X
Description: a cluster of metabolic abnormalities that occur together and significantly increase the risk of cardiovascular diseases, stroke, and diabetes, primarily driven by insulin resistance and central obesity
Cause: insulin resistance, abdominal obesity, genetic predisposition, unhealthy lifestyle (poor diet, physical inactivity), chronic inflammation, hormonal imbalance
Transmission: none
Risk Groups: individuals with obesity, sedentary lifestyle, poor diet, family history of diabetes or heart disease, older adults, people with conditions like Type 2 diabetes or polycystic ovary syndrome
Incubation Period: gradual development over years
Symptoms:
- large waistline (abdominal obesity)
- high blood pressure (often without symptoms)
- high blood sugar (may cause thirst, frequent urination, fatigue)
- high triglycerides
- low HDL cholesterol
- fatigue and weakness (in some cases)
Progression:
1. development of insulin resistance
2. accumulation of abdominal fat and metabolic imbalance
3. appearance of multiple risk factors
4. progression to cardiovascular disease or diabetes if untreated
Common Locations: systemic (affects blood vessels, heart, liver, endocrine system)
Duration: chronic condition
Severity: moderate_to_severe
Complications: coronary heart disease, stroke, Type 2 diabetes, atherosclerosis, fatty liver disease, kidney disease
Home Remedy: balanced diet (low sugar, low saturated fat), regular exercise, weight loss, stress management, adequate sleep
Avoid: sedentary lifestyle, smoking, excessive alcohol, high-calorie processed foods, sugary drinks
When to See a Doctor: if you have risk factors such as high blood pressure, high blood sugar, or increased waist size
Emergency Signs: chest pain, shortness of breath, signs of stroke (weakness, speech difficulty)
Prevention: healthy lifestyle, maintaining normal weight, regular health checkups, monitoring blood pressure, glucose, and cholesterol
Contagious Period: none
Special Notes: diagnosis requires at least three of five criteria (abdominal obesity, high BP, high glucose, high triglycerides, low HDL); early lifestyle changes can significantly reverse or control the condition
===
Disease: Cushing's Syndrome
Aliases: hypercortisolism, Cushing syndrome
Description: a hormonal disorder caused by prolonged exposure to high levels of cortisol, leading to characteristic physical changes, metabolic disturbances, and increased risk of serious health complications
Cause: long-term use of glucocorticoid medications (most common), pituitary tumors (Cushing's disease), adrenal tumors, ectopic ACTH-producing tumors, or abnormal cortisol regulation
Transmission: none
Risk Groups: adults aged 30-50, women (higher prevalence), individuals using long-term corticosteroids, people with conditions like Type 2 diabetes or high blood pressure
Incubation Period: gradual development over months to years
Symptoms:
- weight gain (especially around abdomen, face, and upper back)
- moon face (round face)
- fat accumulation between shoulders (buffalo hump)
- thin arms and legs
- easy bruising
- purple stretch marks (abdomen, thighs, breasts)
- muscle weakness
- fatigue
- high blood pressure
- high blood sugar
- mood changes (depression, anxiety)
- irregular menstrual cycles (in women)
- decreased libido or fertility issues
Progression:
1. prolonged cortisol elevation
2. metabolic and hormonal imbalance
3. visible physical changes and systemic symptoms
4. complications affecting cardiovascular, metabolic, and bone health
Common Locations: systemic (affects endocrine system, metabolism, skin, muscles, cardiovascular system)
Duration: chronic (until treated)
Severity: moderate_to_severe
Complications: heart attack and stroke, blood clots, infections, osteoporosis and fractures, hypertension, insulin resistance and Type 2 diabetes, depression and cognitive issues
Home Remedy: not sufficient alone; supportive lifestyle changes include healthy diet, stress management, and regular monitoring
Avoid: prolonged unnecessary use of corticosteroids, unmanaged stress, high-sugar and high-fat diets
When to See a Doctor: if experiencing rapid weight gain, unusual fat distribution, muscle weakness, or hormonal symptoms
Emergency Signs: severe infection, breathing difficulty, chest pain, or signs of stroke
Prevention: careful use of steroid medications under medical supervision, early detection of hormonal imbalances
Contagious Period: none
Special Notes: diagnosis often requires multiple tests (urine, saliva, blood cortisol levels); treatment depends on the cause and may include surgery, medication, or radiation
===
Disease: High Cholesterol
Aliases: hypercholesterolemia, hyperlipidemia, hyperlipoproteinemia
Description: a condition in which there are high levels of cholesterol in the blood, leading to plaque buildup in arteries (atherosclerosis) and increasing the risk of cardiovascular diseases such as heart attack and stroke
Cause: unhealthy diet (high in saturated and trans fats), lack of physical activity, smoking, excessive alcohol consumption, stress-related hormonal changes, genetic conditions, associated diseases such as Type 2 diabetes, chronic kidney disease, or hormonal disorders
Transmission: none
Risk Groups: older adults, individuals with family history of high cholesterol, people with obesity, smokers, individuals with sedentary lifestyle, patients with conditions like Type 2 diabetes
Incubation Period: develops gradually over years
Symptoms:
- usually no obvious symptoms
- may be detected through blood tests
- chest pain (angina) in severe cases
- fatigue in severe cases
- signs of cardiovascular disease in severe cases
Progression:
1. increased LDL (bad cholesterol) or decreased HDL (good cholesterol)
2. plaque formation in arteries (atherosclerosis)
3. narrowing or blockage of blood vessels
4. complications such as heart attack or stroke
Common Locations: blood vessels and arteries (especially coronary arteries, brain arteries, peripheral arteries)
Duration: chronic
Severity: mild_to_severe
Complications: atherosclerosis, coronary artery disease, heart attack, stroke, peripheral arterial disease, angina
Home Remedy: heart-healthy diet (low saturated fats, high fiber), regular exercise, weight management, quitting smoking, limiting alcohol intake
Avoid: high-fat processed foods, excessive sugar and alcohol, sedentary lifestyle, smoking
When to See a Doctor: routine screening (especially after age 20), family history of heart disease, symptoms of heart or vascular problems
Emergency Signs: chest pain or pressure, shortness of breath, sudden weakness or numbness (possible stroke)
Prevention: balanced diet and physical activity, regular cholesterol screening, maintaining healthy weight, managing related conditions
Contagious Period: none
Special Notes: includes different types (LDL bad cholesterol, HDL good cholesterol, and VLDL); diagnosis is primarily through blood lipid profile tests; treatment may include lifestyle changes and medications such as statins
===
Disease: Osteoporosis
Aliases: porous bone disease, bone density loss disorder
Description: a progressive bone disorder in which bone mineral density and bone mass decrease, leading to fragile bones that are more prone to fractures, often without noticeable symptoms until a break occurs
Cause: age-related bone loss, hormonal changes (especially low estrogen after menopause and low testosterone in men), low calcium and vitamin D intake, lack of physical activity, long-term use of medications (e.g., corticosteroids), medical conditions such as Hyperthyroidism, Cushing's syndrome, rheumatoid arthritis, lifestyle factors (smoking, excessive alcohol use), genetic predisposition
Transmission: none
Risk Groups: postmenopausal women, older adults (especially over 50 years), individuals with low body weight, people with family history of osteoporosis, individuals with sedentary lifestyle, patients on long-term steroid therapy
Incubation Period: develops slowly over years
Symptoms:
- usually no symptoms (silent disease)
- fractures after minor falls or stress
- back pain (due to vertebral fractures)
- loss of height over time
- stooped posture (kyphosis)
Progression:
1. gradual loss of bone density
2. weakening of bone structure
3. increased fragility
4. fractures (hip, spine, wrist most common)
Common Locations: bones of hip, spine (vertebrae), wrist, and other weight-bearing bones
Duration: chronic
Severity: moderate_to_severe
Complications: frequent fractures, chronic pain (especially spinal fractures), reduced mobility and disability, increased risk of mortality after hip fractures
Home Remedy: calcium-rich diet (milk, leafy greens), adequate vitamin D (sunlight, supplements), regular weight-bearing exercise (walking, strength training)
Avoid: smoking, excessive alcohol, sedentary lifestyle, poor nutrition
When to See a Doctor: after minor fractures, signs of height loss or posture changes, risk factors like menopause or long-term steroid use
Emergency Signs: sudden severe back pain (possible spinal fracture), inability to move after a fall
Prevention: regular exercise, balanced diet rich in calcium and vitamin D, maintaining healthy weight, avoiding smoking and excessive alcohol
Contagious Period: none
Special Notes: often diagnosed using bone density tests (DEXA scan); early detection is important to prevent fractures; lifestyle changes and medications can slow progression
===
Disease: Vitamin D Deficiency
Aliases: hypovitaminosis D
Description: a condition in which the body does not have enough vitamin D to maintain healthy bones and normal body functions, leading to impaired calcium absorption, weakened bones, and potential systemic effects
Cause: inadequate dietary intake of vitamin D, insufficient sunlight exposure, malabsorption disorders, liver or kidney disease affecting vitamin D activation, medications interfering with vitamin D metabolism, obesity (vitamin D sequestration in fat tissue)
Transmission: none
Risk Groups: infants (especially breastfed without supplementation), older adults, people with dark skin, individuals with limited sun exposure, people with obesity, patients with chronic kidney or liver disease, individuals with gastrointestinal disorders affecting absorption
Incubation Period: gradual onset over months to years
Symptoms:
- often mild or no early symptoms
- fatigue
- bone pain
- muscle weakness
- frequent fractures
- delayed growth or bone deformities in children
Progression:
1. low vitamin D levels
2. reduced calcium absorption
3. decreased bone mineralization
4. development of bone disorders
Common Locations: bones, muscles, immune and nervous systems
Duration: chronic (until corrected)
Severity: mild_to_severe
Complications: Osteoporosis, fractures, Rickets (in children), Osteomalacia (in adults)
Home Remedy: safe sunlight exposure, vitamin D-rich diet (milk, milk products, fortified foods), supplements (as prescribed)
Avoid: prolonged lack of sun exposure, poor diet lacking vitamin D, excessive reliance on unverified supplements
When to See a Doctor: persistent fatigue or bone pain, frequent fractures, high-risk individuals (elderly, chronic illness)
Emergency Signs: severe bone weakness leading to fractures, symptoms of vitamin D toxicity (nausea, confusion, irregular heartbeat)
Prevention: adequate sunlight exposure, balanced diet with vitamin D, supplementation for high-risk groups
Contagious Period: none
Special Notes: diagnosed through blood tests measuring vitamin D levels; both deficiency and excess (toxicity) can be harmful, so proper dosing is important
===
Disease: Vitamin B12 Deficiency Anemia
Aliases: cobalamin deficiency anemia, megaloblastic anemia B12-related
Description: a condition in which the body cannot produce enough healthy red blood cells due to a lack of vitamin B12, leading to impaired oxygen transport and potential damage to the nervous system
Cause: inadequate intake of vitamin B12 (strict vegan or poor diet), impaired absorption due to Pernicious anemia (lack of intrinsic factor), gastrointestinal disorders such as Crohn's disease, Ulcerative colitis, stomach or intestinal surgery, chronic alcohol consumption, medications (e.g., metformin, proton pump inhibitors), infections such as Helicobacter pylori infection
Transmission: none
Risk Groups: older adults, vegetarians and vegans, people with autoimmune diseases, individuals with gastrointestinal disorders, patients with history of stomach or intestinal surgery, people with chronic alcohol use
Incubation Period: gradual onset over months to years
Symptoms:
- fatigue, weakness
- pale skin
- shortness of breath
- dizziness, headaches
- tingling or numbness (paresthesia)
- difficulty walking
- confusion, memory loss
- depression or irritability
- vision problems
- diarrhea and weight loss
- glossitis (smooth, red tongue)
Progression:
1. vitamin B12 deficiency
2. abnormal red blood cell formation (megaloblastic changes)
3. anemia develops
4. neurological complications may occur if untreated
Common Locations: blood, bone marrow, nervous system
Duration: chronic if untreated
Severity: moderate_to_severe
Complications: neurological damage (may be permanent), cognitive decline or dementia-like symptoms, infertility, increased risk of infections and bleeding, developmental issues in infants, increased risk of stomach cancer (with pernicious anemia)
Home Remedy: vitamin B12-rich diet (meat, fish, eggs, dairy, fortified foods), supplementation (oral or injectable as prescribed)
Avoid: prolonged poor diet lacking B12, excessive alcohol consumption, ignoring early neurological symptoms
When to See a Doctor: persistent fatigue or weakness, numbness or tingling, memory or mood changes, unexplained anemia
Emergency Signs: severe neurological symptoms (loss of coordination, confusion), extreme weakness or breathlessness
Prevention: balanced diet with adequate vitamin B12, supplements for high-risk groups (vegans, elderly), regular screening in at-risk individuals
Contagious Period: none
Special Notes: diagnosed via blood tests (hemoglobin, vitamin B12 levels); early treatment is crucial to prevent irreversible nerve damage; treatment may require lifelong supplementation in some cases
===
Disease: Gout
Aliases: gouty arthritis, urate crystal arthritis
Description: a painful form of arthritis caused by the buildup of uric acid crystals in and around joints, leading to sudden episodes (flares) of intense pain, swelling, and redness
Cause: excess uric acid (hyperuricemia) in the blood, reduced excretion of uric acid by kidneys, high intake of purine-rich foods (red meat, seafood), alcohol consumption (especially beer and spirits), certain medications (diuretics, beta-blockers, chemotherapy drugs)
Transmission: none
Risk Groups: adult men (more common than women), people with obesity, individuals with high blood pressure, diabetes, or kidney disease, those with a family history of gout, people consuming high-purine diets or excessive alcohol
Incubation Period: none (chronic condition with acute flares)
Symptoms:
- sudden severe joint pain (often at night)
- swelling and tenderness in joints
- redness and warmth over the joint
- limited joint movement
- most commonly affects the big toe, but also ankles, knees, fingers
Progression:
1. elevated uric acid levels (often asymptomatic)
2. crystal formation in joints
3. acute gout attack (painful flare)
4. recurrent flares if untreated
5. chronic gout with joint damage and tophi formation
Common Locations: big toe (most common), ankles, knees, fingers, wrists
Duration: acute flare 3-10 days; chronic condition with recurring episodes
Severity: moderate_to_severe
Complications: joint deformity and permanent damage, kidney stones, tophi (urate crystal deposits under the skin), reduced mobility, associated conditions like Hypertension and Type 2 Diabetes
Home Remedy: rest and elevate affected joint, apply ice packs (wrapped), stay well hydrated, maintain healthy weight, limit purine-rich foods and alcohol
Avoid: red meat, organ meats, seafood, sugary drinks and high-fructose foods, alcohol (especially beer), dehydration
When to See a Doctor: first episode of severe joint pain, worsening or persistent symptoms, medications not relieving symptoms
Emergency Signs: fever with joint pain (possible infection like septic arthritis), severe swelling and inability to move joint
Prevention: maintain healthy diet and weight, regular exercise, adequate hydration, long-term urate-lowering therapy (e.g., allopurinol) if prescribed
Contagious Period: none
Special Notes: diagnosed using joint fluid analysis, blood tests (uric acid), and imaging; early treatment helps prevent chronic joint damage; lifelong management may be required to prevent flares
===
Disease: Polycystic Ovary Syndrome
Aliases: PCOS, Stein-Leventhal Syndrome
Description: a hormonal and metabolic disorder affecting the ovaries, characterized by irregular ovulation, excess androgen (male hormone) levels, and multiple small fluid-filled sacs in the ovaries, often impacting fertility and overall health
Cause: hormonal imbalance (elevated androgens), insulin resistance, genetic predisposition (family history), metabolic dysfunction
Transmission: none
Risk Groups: females of reproductive age (often starting at puberty), individuals with family history of PCOS, people with obesity or insulin resistance
Incubation Period: none (chronic condition developing over time)
Symptoms:
- irregular or missed menstrual periods
- excess facial or body hair (hirsutism)
- acne and oily skin
- weight gain or difficulty losing weight
- thinning hair or hair loss
- darkened skin patches (acanthosis nigricans)
- infertility or difficulty conceiving
Progression:
1. hormonal imbalance begins
2. irregular ovulation or anovulation
3. development of metabolic and reproductive symptoms
4. long-term complications if untreated
Common Locations: ovaries, endocrine system
Duration: lifelong condition (manageable but not curable)
Severity: mild_to_moderate
Complications: Type 2 Diabetes, Hypertension, Heart disease, high LDL and low HDL cholesterol, infertility, Sleep apnea, depression and anxiety
Home Remedy: maintain healthy weight, regular physical activity, balanced diet (low sugar, high fiber), stress management
Avoid: sedentary lifestyle, high-sugar and processed foods, excessive weight gain
When to See a Doctor: irregular or absent periods, difficulty getting pregnant, excessive hair growth or severe acne, unexplained weight gain
Emergency Signs: none specific (but severe complications require medical care)
Prevention: not fully preventable (genetic and hormonal factors); healthy lifestyle can reduce severity and complications
Contagious Period: none
Special Notes: diagnosed through symptoms, hormone tests, and ultrasound; treatment focuses on symptom management (lifestyle, hormonal therapy, medications); early management helps prevent long-term complications
===
Disease: Gastritis
Aliases: stomach lining inflammation
Description: inflammation or swelling of the stomach lining that can be acute (short-term) or chronic (long-term), often leading to discomfort, digestive issues, and sometimes complications like bleeding
Cause: long-term use of NSAIDs (aspirin, ibuprofen, naproxen), excessive alcohol consumption, infection with Helicobacter pylori infection, autoimmune disorders such as Pernicious anemia, bile reflux, severe stress, trauma, or major illness, viral infections (e.g., cytomegalovirus, herpes simplex virus), corrosive substances or drug use
Transmission: not directly contagious (except H. pylori infection in some cases)
Risk Groups: people using painkillers (NSAIDs) frequently, heavy alcohol users, older adults, individuals with weakened immune systems, people with autoimmune conditions
Incubation Period: varies depending on cause
Symptoms:
- upper abdominal pain or discomfort
- nausea and vomiting
- loss of appetite
- bloating
- black or tarry stools in severe cases
- vomiting blood or coffee-ground-like material in severe cases
Progression:
1. irritation of stomach lining
2. inflammation develops
3. acute or chronic gastritis
4. possible complications (ulcers, bleeding, cancer risk)
Common Locations: stomach lining (gastric mucosa)
Duration: acute (short-term, days to weeks); chronic (months to years)
Severity: mild_to_severe
Complications: stomach ulcers, internal bleeding, anemia, increased risk of gastric cancer
Home Remedy: avoid alcohol and irritant foods, eat smaller balanced meals, manage stress, stay hydrated
Avoid: NSAIDs without medical guidance, alcohol and smoking, spicy, acidic, or irritating foods
When to See a Doctor: persistent abdominal pain, ongoing nausea or vomiting, symptoms lasting more than a few days
Emergency Signs: vomiting blood, black or tarry stools, severe abdominal pain
Prevention: limit NSAID use, reduce alcohol intake, treat H. pylori infection early, maintain healthy diet and lifestyle
Contagious Period: none (except possible transmission of H. pylori)
Special Notes: diagnosed using endoscopy, blood tests, and H. pylori tests; treatment includes acid-reducing medicines (antacids, PPIs, H2 blockers) and antibiotics if infection is present; prognosis is usually good with proper treatment
===
Disease: Peptic Ulcer Disease
Aliases: gastric ulcer, duodenal ulcer
Description: a condition where open sores (ulcers) develop in the lining of the stomach or the upper part of the small intestine (duodenum), usually due to damage from stomach acid
Cause: infection with Helicobacter pylori infection, long-term use of NSAIDs (aspirin, ibuprofen, naproxen), excessive stomach acid production, rare conditions like Zollinger-Ellison syndrome, smoking and alcohol use (worsening factors)
Transmission: not directly contagious (except H. pylori in some cases)
Risk Groups: people taking NSAIDs regularly, smokers and alcohol users, individuals infected with H. pylori, older adults
Incubation Period: varies (depends on cause)
Symptoms:
- burning or gnawing stomach pain (common)
- pain between meals or at night
- bloating and fullness
- nausea
- loss of appetite
- weight loss
- black or bloody stools in severe cases
- vomiting blood in severe cases
- fatigue in severe cases
Progression:
1. damage to stomach or duodenal lining
2. ulcer formation
3. recurring pain episodes
4. complications if untreated
Common Locations: stomach (gastric ulcer), duodenum (duodenal ulcer)
Duration: healing typically within 6-8 weeks with treatment; may recur if underlying cause persists
Severity: moderate_to_severe
Complications: internal bleeding, perforation (hole in stomach or intestine), blockage of digestive tract, Peritonitis (if rupture occurs)
Home Remedy: eat balanced meals at regular times, avoid trigger foods (spicy, fatty, caffeine), stop smoking and limit alcohol, manage stress
Avoid: NSAIDs without medical advice, alcohol, smoking, late-night heavy meals, irritant foods
When to See a Doctor: persistent stomach pain, unexplained weight loss, symptoms lasting more than a few days
Emergency Signs: vomiting blood, black or tarry stools, severe abdominal pain (possible perforation)
Prevention: treat H. pylori infection early, limit NSAID use, avoid smoking and excessive alcohol, maintain healthy diet
Contagious Period: none (except possible H. pylori spread)
Special Notes: diagnosed via endoscopy, biopsy, and H. pylori tests; treated with antibiotics (for H. pylori) and acid-reducing medicines (PPIs, H2 blockers); most ulcers heal completely with proper treatment
===
Disease: Gastroesophageal Reflux Disease
Aliases: GERD, chronic acid reflux, reflux disease
Description: a long-term condition in which stomach contents repeatedly flow back into the esophagus, causing symptoms like heartburn and potential complications over time
Cause: weakening or improper relaxation of the lower esophageal sphincter (LES), increased abdominal pressure (obesity, pregnancy), smoking or exposure to secondhand smoke, certain medications (NSAIDs, calcium channel blockers, sedatives, antidepressants)
Transmission: not contagious
Risk Groups: overweight or obese individuals, pregnant women, smokers, people taking certain medications
Incubation Period: none (develops gradually over time)
Symptoms:
- heartburn (burning chest pain)
- regurgitation (acid or food coming back into throat or mouth)
- chest pain
- nausea
- difficulty swallowing
- chronic cough or hoarseness
- vomiting blood in severe cases
- black or tarry stools in severe cases
- unexplained weight loss in severe cases
Progression:
1. occasional acid reflux (GER)
2. frequent reflux episodes
3. chronic irritation of esophagus
4. complications if untreated
Common Locations: esophagus (mainly lower esophagus)
Duration: chronic (long-term condition); symptoms may be persistent or recurrent
Severity: mild_to_severe
Complications: Esophagitis, Esophageal stricture, Barrett's esophagus, increased risk of esophageal cancer
Home Remedy: maintain healthy weight, elevate head while sleeping, avoid trigger foods (fatty, spicy, caffeine), eat smaller meals, avoid lying down after eating
Avoid: smoking, alcohol, large or late-night meals, trigger medications (if possible, under doctor guidance)
When to See a Doctor: frequent or persistent heartburn, symptoms not improving with OTC medicines, difficulty swallowing or chest pain
Emergency Signs: vomiting blood, black or tarry stools, severe chest pain
Prevention: maintain healthy weight, avoid smoking and alcohol, follow healthy diet and lifestyle, manage underlying conditions
Contagious Period: none
Special Notes: diagnosed mainly through symptoms; tests include endoscopy and pH monitoring; treated with antacids, H2 blockers, and proton pump inhibitors (PPIs); surgery (fundoplication) may be needed in severe cases
===
Disease: Irritable Bowel Syndrome
Aliases: IBS, spastic colon, irritable colon
Description: a common, long-term disorder affecting the digestive system, characterized by recurring abdominal discomfort and changes in bowel habits without any visible structural damage
Cause: exact cause unknown; possible factors include abnormal gut-brain interaction, intestinal muscle spasms, stress and psychological factors, food sensitivities or intolerances, changes in gut bacteria, previous infections (post-infectious IBS)
Transmission: not contagious
Risk Groups: young adults (especially under 50), women (more commonly affected), people with stress, anxiety, or depression, individuals with family history of IBS
Incubation Period: none (chronic condition)
Symptoms:
- abdominal pain or cramping (often relieved after bowel movement)
- diarrhoea, constipation, or both (alternating)
- bloating and abdominal swelling
- excess gas (flatulence)
- urgency to pass stool
- fatigue (less common)
- nausea (less common)
- heartburn (less common)
Progression:
1. intermittent digestive discomfort
2. recurring bowel habit changes
3. chronic symptom cycles (flare-ups and remission)
4. symptoms triggered by stress or diet
Common Locations: large intestine (colon), small intestine
Duration: lifelong condition; symptoms fluctuate over time
Severity: mild_to_moderate
Complications: reduced quality of life, anxiety and depression, social and lifestyle limitations, does not cause cancer or permanent bowel damage
Home Remedy: manage stress (yoga, meditation), follow a balanced fiber-controlled diet, identify and avoid trigger foods, eat smaller regular meals, stay hydrated
Avoid: high-fat or spicy foods, caffeine and alcohol, large meals, stress triggers
When to See a Doctor: persistent symptoms affecting daily life, uncertainty about diagnosis, need for symptom management
Emergency Signs: unexplained weight loss, blood in stool, persistent vomiting, symptoms starting after age 50
Prevention: no definite prevention; can be managed with diet, stress control, and healthy lifestyle
Contagious Period: none
Special Notes: diagnosis is based on symptoms and ruling out other conditions; treatment focuses on symptom control (diet changes, medications, stress management); common triggers include stress and certain foods
===
Disease: Crohn's Disease
Aliases: regional enteritis, regional ileitis
Description: a chronic (long-term) condition that causes inflammation anywhere in the digestive tract, most commonly affecting the small intestine and the beginning of the large intestine
Cause: exact cause unknown; possible factors include autoimmune reaction (immune system attacks digestive tract), genetic predisposition, environmental factors, abnormal gut microbiome
Transmission: not contagious
Risk Groups: people with family history of Crohn's disease, smokers, individuals aged 15-35 (common onset age), people taking certain medications (NSAIDs, antibiotics, birth-control pills), those with high-fat diets
Incubation Period: none (chronic disease)
Symptoms:
- persistent diarrhea
- abdominal pain and cramping
- weight loss
- fatigue
- fever
- loss of appetite
- anemia
- joint pain (extra-intestinal symptom)
- eye inflammation (extra-intestinal symptom)
- skin problems (extra-intestinal symptom)
Progression:
1. inflammation in digestive tract
2. recurring flare-ups (active disease)
3. periods of remission (few or no symptoms)
4. possible complications over time
Common Locations: small intestine (especially ileum), beginning of large intestine (colon), can affect any part from mouth to anus
Duration: lifelong condition; cycles of flare-ups and remission
Severity: moderate_to_severe
Complications: intestinal obstruction (blockage), fistulas (abnormal connections), abscesses (infection pockets), anal fissures, ulcers in digestive tract, malnutrition, inflammation in joints, eyes, or skin
Home Remedy: eat small, frequent meals, maintain balanced nutrition, keep a food diary to identify triggers, stay hydrated, manage stress
Avoid: smoking, high-fat or trigger foods, carbonated drinks, large meals
When to See a Doctor: persistent diarrhea or abdominal pain, unexplained weight loss, blood in stool, prolonged fatigue or fever
Emergency Signs: severe abdominal pain, signs of intestinal blockage, heavy bleeding, high fever with complications
Prevention: no known prevention; risk can be reduced by avoiding smoking and maintaining a healthy lifestyle
Contagious Period: none
Special Notes: no cure, but treatable with medications, diet changes, and sometimes surgery; diagnosis involves blood tests, stool tests, imaging, and endoscopy; symptoms may resemble IBS but Crohn's causes actual inflammation and damage to the intestine
===
Disease: Ulcerative Colitis
Aliases: UC
Description: a chronic condition that causes inflammation and ulcers in the inner lining of the colon (large intestine) and rectum, often starting in the rectum and spreading upward
Cause: exact cause unknown; possible factors include immune system dysfunction (autoimmune response), genetic predisposition, imbalance in gut bacteria, environmental influences (diet, antibiotics, pollution)
Transmission: not contagious
Risk Groups: people aged 15-30 (common onset), individuals with family history of UC, all ethnic groups
Incubation Period: none (chronic condition)
Symptoms:
- diarrhea (often with blood, mucus, or pus)
- abdominal pain and cramping
- urgency to pass stool
- rectal bleeding
- fatigue
- weight loss
- fever in severe cases
- delayed growth in children
Progression:
1. inflammation begins in rectum
2. spreads through colon
3. flare-ups (active symptoms) and remission cycles
4. complications if untreated
Common Locations: rectum, colon (large intestine)
Duration: lifelong condition; alternating periods of flare-ups and remission
Severity: mild_to_severe
Complications: severe bleeding, anemia, dehydration, Osteoporosis, toxic megacolon, increased risk of colon cancer, inflammation of joints, skin, and eyes, blood clots
Home Remedy: eat small, frequent meals, maintain a balanced diet, keep a food diary to identify triggers, stay hydrated, manage stress (yoga, relaxation techniques)
Avoid: trigger foods (spicy, fatty, high-fiber during flares), alcohol, smoking, unmanaged stress
When to See a Doctor: persistent diarrhea or blood in stool, abdominal pain lasting long, unexplained weight loss, frequent bowel urgency
Emergency Signs: severe abdominal pain, heavy bleeding, high fever, signs of toxic megacolon
Prevention: no known prevention; early diagnosis and management reduce complications
Contagious Period: none
Special Notes: no cure, but treatable with medications (aminosalicylates, steroids, immunosuppressants, biologics); surgery (colectomy) may be required in severe cases; diagnosis includes colonoscopy, stool tests, and blood tests
===
Disease: Cirrhosis
Aliases: liver cirrhosis, end-stage liver disease
Description: a progressive condition where long-term liver damage causes scarring (fibrosis), replacing healthy liver tissue and impairing liver function; represents the final stage of many chronic liver diseases
Cause: chronic liver damage due to Hepatitis B and Hepatitis C, long-term alcohol misuse, metabolic-associated fatty liver disease linked to obesity and diabetes; less common causes include autoimmune hepatitis, bile duct disorders, genetic/inherited liver diseases, certain medications
Transmission: not directly contagious; underlying causes like hepatitis B/C may be infectious
Risk Groups: heavy alcohol users, people with chronic hepatitis infections, individuals with obesity, diabetes, or high cholesterol, people with family history of liver disease
Incubation Period: varies (develops slowly over years of liver damage)
Symptoms:
- fatigue
- loss of appetite
- nausea
- weight loss
- mild abdominal discomfort (early)
- jaundice (yellowing of skin or eyes) in late stage
- swelling in legs (edema) in late stage
- abdominal fluid buildup (ascites) in late stage
- easy bruising and bleeding in late stage
- confusion or memory problems (hepatic encephalopathy) in late stage
- spider-like blood vessels on skin in late stage
Progression:
1. chronic liver injury
2. inflammation and fibrosis
3. extensive scarring (cirrhosis)
4. liver failure (end-stage disease)
Common Locations: liver
Duration: lifelong condition; progresses over years
Severity: severe
Complications: portal hypertension, ascites and infections, varices (enlarged veins) with bleeding, Hepatocellular carcinoma, kidney failure, hepatic encephalopathy
Home Remedy: avoid alcohol completely, follow a balanced low-salt diet, maintain healthy weight, regular exercise, manage diabetes and blood pressure
Avoid: alcohol, high-salt and fatty foods, unnecessary medications or supplements without doctor advice
When to See a Doctor: persistent fatigue, jaundice, or abdominal swelling, unexplained weight loss, symptoms of liver disease
Emergency Signs: vomiting blood, black or tarry stools, severe confusion, sudden abdominal swelling or pain, high fever
Prevention: limit or avoid alcohol, vaccination for hepatitis B, prevent hepatitis C (safe practices, avoid needle sharing), maintain healthy weight and lifestyle
Contagious Period: none (except underlying infections like hepatitis)
Special Notes: damage is usually irreversible; early treatment can slow progression; managed with medications, lifestyle changes, and treating underlying cause; advanced cases may require liver transplant
===
Disease: Fatty Liver Disease
Aliases: nonalcoholic fatty liver disease, NAFLD, alcohol-associated liver disease, ALD, metabolic dysfunction-associated steatotic liver disease, MASLD
Description: a condition where excess fat accumulates in liver cells, which may or may not cause inflammation; in advanced stages, it can lead to liver damage, scarring, and serious complications
Cause: fat accumulation in liver due to metabolic factors (obesity, insulin resistance, type 2 diabetes), high cholesterol or triglycerides, alcohol-related causes (heavy and long-term alcohol consumption), other contributing factors such as certain medications, rapid weight loss, infections such as Hepatitis C, exposure to toxins
Transmission: not contagious
Risk Groups: people with obesity or overweight, individuals with type 2 diabetes or prediabetes, people with high cholesterol or triglycerides, middle-aged or older adults, heavy alcohol users
Incubation Period: develops gradually over years
Symptoms:
- usually asymptomatic (silent disease) early stage
- fatigue (if present)
- mild discomfort in upper right abdomen (if present)
- jaundice in advanced stages
- abdominal swelling in advanced stages
- weakness in advanced stages
Progression:
1. fat accumulation in liver (simple fatty liver)
2. inflammation and cell damage (steatohepatitis)
3. fibrosis (scarring)
4. progression to Cirrhosis
5. possible liver failure or cancer
Common Locations: liver
Duration: chronic (long-term condition)
Severity: mild_to_severe
Complications: liver fibrosis, cirrhosis, Liver cancer, liver failure
Home Remedy: maintain healthy weight, follow a balanced diet (low sugar, low fat), regular exercise, control diabetes, cholesterol, and blood pressure
Avoid: alcohol (especially in any fatty liver condition), sugary foods and drinks, high-fat processed foods, unnecessary medications or supplements
When to See a Doctor: persistent fatigue or abdominal discomfort, abnormal liver test results, risk factors like obesity or diabetes
Emergency Signs: yellowing of skin or eyes (jaundice), severe abdominal swelling, confusion or bleeding
Prevention: maintain healthy weight, regular physical activity, balanced diet, limit or avoid alcohol, manage metabolic conditions
Contagious Period: none
Special Notes: often reversible in early stages with lifestyle changes; no specific approved medications for most cases; early detection is important to prevent progression to severe liver disease
===
Disease: Gallstones
Aliases: cholelithiasis, gallstone disease
Description: a condition where hardened deposits (stones), usually made of cholesterol or bilirubin, form in the gallbladder; often asymptomatic but can cause severe pain and complications if they block bile ducts
Cause: imbalance in bile composition (excess cholesterol in bile, excess bilirubin production), poor gallbladder emptying (bile becomes concentrated), associated conditions (Cirrhosis, certain blood disorders), lifestyle factors (obesity, high-fat diet, rapid weight loss)
Transmission: not contagious
Risk Groups: women (especially after pregnancy), people over 40 years, overweight or obese individuals, people with diabetes, those with family history of gallstones, individuals on estrogen therapy
Incubation Period: varies (develops gradually over time)
Symptoms:
- many people are asymptomatic
- sudden severe pain in upper right abdomen (biliary colic)
- pain lasting 1-5 hours
- pain radiating to back or right shoulder
- nausea and vomiting
- persistent abdominal pain (complication symptom)
- fever (complication symptom)
- jaundice (complication symptom)
Progression:
1. imbalance in bile composition
2. crystal formation
3. stone development in gallbladder
4. blockage of bile duct (symptoms begin)
5. complications if untreated
Common Locations: gallbladder, bile ducts
Duration: chronic condition; symptoms occur in episodes (flares)
Severity: mild_to_severe
Complications: Cholecystitis, bile duct blockage, Pancreatitis, jaundice, rare: gallbladder cancer
Home Remedy: maintain healthy weight, eat balanced low-fat diet, regular meals (avoid fasting), increase fiber intake
Avoid: high-fat and high-cholesterol foods, rapid weight loss, skipping meals
When to See a Doctor: repeated abdominal pain episodes, nausea with pain, symptoms interfering with daily life
Emergency Signs: severe abdominal pain that doesn't go away, fever with chills, yellowing of skin or eyes, persistent vomiting
Prevention: maintain healthy weight, gradual weight loss if needed, high-fiber diet, regular physical activity
Contagious Period: none
Special Notes: many cases require no treatment if asymptomatic; symptomatic cases often treated with gallbladder removal (laparoscopic cholecystectomy); life without a gallbladder is generally normal
===
Disease: Pancreatitis
Aliases: pancreatic inflammation
Description: a condition where the pancreas becomes inflamed due to premature activation of digestive enzymes, causing the pancreas to damage itself; can be acute (short-term) or chronic (long-term)
Cause: most common causes: Gallstones blocking pancreatic ducts, heavy alcohol use; other causes: high triglyceride levels, high calcium levels, certain medications, genetic disorders, infections or trauma, pancreatic cancer, smoking
Transmission: not contagious
Risk Groups: alcohol users, individuals with gallstones, obese individuals, people with high triglycerides, smokers, individuals with family history of pancreatic disease
Incubation Period: varies (depends on cause; can be sudden in acute cases)
Symptoms:
- severe upper abdominal pain (radiates to back) in acute pancreatitis
- nausea and vomiting in acute pancreatitis
- fever in acute pancreatitis
- rapid heart rate in acute pancreatitis
- persistent abdominal pain in chronic pancreatitis
- weight loss in chronic pancreatitis
- diarrhea in chronic pancreatitis
- greasy or oily stools (steatorrhea) in chronic pancreatitis
- high blood sugar in chronic pancreatitis
- jaundice in chronic pancreatitis
Progression:
1. enzyme activation inside pancreas
2. inflammation and tissue damage
3. acute episode or repeated damage
4. chronic inflammation
5. permanent pancreatic damage
Common Locations: pancreas
Duration: acute (few days); chronic (months to years)
Severity: moderate_to_severe
Complications: pancreatic necrosis (tissue death), infection or abscess, pseudocysts, diabetes, Pancreatic cancer, organ failure (severe cases)
Home Remedy: follow a low-fat diet, stay hydrated, avoid alcohol completely, stop smoking, maintain healthy weight
Avoid: alcohol, smoking, fatty and heavy meals, untreated gallstones
When to See a Doctor: severe or persistent abdominal pain, repeated vomiting, unexplained weight loss, jaundice
Emergency Signs: severe abdominal pain radiating to back, high fever, rapid heartbeat, confusion or weakness, inability to keep fluids down
Prevention: limit or avoid alcohol, treat gallstones early, maintain healthy weight, control triglyceride levels, avoid smoking
Contagious Period: none
Special Notes: acute pancreatitis can be life-threatening and may require hospitalization; chronic pancreatitis can lead to permanent damage and diabetes; treatment focuses on managing cause, pain relief, and supporting digestion
===
Disease: Appendicitis
Aliases: inflamed appendix, acute appendicitis, appendix inflammation
Description: a condition where the appendix (a small pouch attached to the large intestine) becomes inflamed, swollen, and filled with pus; it is a medical emergency that usually requires surgical removal
Cause: blockage of the appendix (most common): hardened stool, swollen lymph tissue; bacterial infection leading to inflammation; less common triggers: parasites or tumors
Transmission: not contagious
Risk Groups: most common in ages 10-30, slightly more common in males, can occur at any age
Incubation Period: rapid onset (symptoms develop within hours to 1-2 days)
Symptoms:
- pain around the belly button (early symptom)
- mild cramp-like pain (early symptom)
- sharp pain in lower right abdomen (later classic sign)
- pain worsening with movement, coughing, or walking
- nausea and vomiting
- loss of appetite
- low-grade fever
- constipation or diarrhea
Progression:
1. blockage of appendix
2. bacterial growth and inflammation
3. swelling and increasing pain
4. possible rupture (burst appendix) if untreated
Common Locations: appendix (lower right abdomen)
Duration: acute condition (progresses quickly over hours to days)
Severity: severe
Complications: rupture (burst appendix), Peritonitis, abscess formation (pus collection), widespread infection (sepsis)
Home Remedy: none (requires urgent medical care)
Avoid: delaying medical treatment, self-medicating severe abdominal pain
When to See a Doctor: worsening abdominal pain, pain shifting to lower right abdomen, nausea, fever, or loss of appetite
Emergency Signs: sudden severe pain spreading across abdomen, high fever, persistent vomiting, inability to move due to pain
Prevention: no guaranteed prevention
Contagious Period: none
Special Notes: treated with antibiotics and usually surgery (appendectomy); laparoscopic (keyhole) surgery is most common; recovery is usually quick (1-2 weeks for mild cases); untreated appendicitis can be life-threatening due to rupture
===
Disease: Hemorrhoids
Aliases: piles
Description: a condition in which veins in the rectum or around the anus become swollen and inflamed, leading to discomfort, bleeding, or lumps
Cause: increased pressure in rectal or anal veins, straining during bowel movements (constipation or diarrhea), prolonged sitting on the toilet, obesity, pregnancy, heavy lifting
Transmission: not contagious
Risk Groups: pregnant women, people with chronic constipation or diarrhea, overweight or obese individuals, people with low-fiber diets, individuals who sit for long periods
Incubation Period: none (develops gradually with pressure over time)
Symptoms:
- bleeding during bowel movements (bright red blood)
- itching or irritation around anus
- pain or discomfort, especially during bowel movements
- swelling or lump near the anus
- internal hemorrhoids: painless bleeding
- external hemorrhoids: pain, itching, swelling
- prolapsed hemorrhoids: lump protruding from anus
- thrombosed hemorrhoids: severe pain due to clot
Progression:
1. increased pressure in veins
2. swelling and inflammation
3. formation of internal or external hemorrhoids
4. possible prolapse or clot formation
Common Locations: inside rectum (internal hemorrhoids), around anus (external hemorrhoids)
Duration: mild cases (a few days to weeks); recurrent if risk factors persist
Severity: mild_to_moderate
Complications: thrombosis (painful blood clot), chronic bleeding leading to anemia, prolapse (hemorrhoid protrusion)
Home Remedy: warm sitz baths, high-fiber diet (fruits, vegetables, whole grains), increased fluid intake, cold compress or ice packs, topical creams (e.g., witch hazel, hydrocortisone)
Avoid: straining during bowel movements, sitting too long on the toilet, low-fiber diet, heavy lifting
When to See a Doctor: persistent pain or swelling, bleeding that continues or worsens, symptoms not improving within a week
Emergency Signs: heavy rectal bleeding, severe pain (possible thrombosed hemorrhoid), signs of anemia (fatigue, dizziness)
Prevention: eat a high-fiber diet, drink plenty of water, exercise regularly, avoid straining, respond promptly to bowel urges
Contagious Period: none
Special Notes: diagnosed via physical exam, anoscopy, or sigmoidoscopy; most cases resolve with lifestyle changes; procedures like rubber band ligation or surgery may be needed for severe cases
===
Disease: Constipation
Aliases: chronic constipation (when long-term)
Description: a condition in which bowel movements become infrequent, difficult, or incomplete, often involving hard, dry stools due to slow movement through the colon
Cause: low dietary fiber intake, inadequate fluid intake, lack of physical activity, ignoring the urge to pass stool, medications (opioids, antidepressants, antacids, iron supplements), hormonal or metabolic conditions (e.g., diabetes, thyroid disorders), pelvic floor dysfunction, neurological disorders
Transmission: not contagious
Risk Groups: older adults, women (especially during pregnancy), people with sedentary lifestyles, individuals with low-fiber diets, patients on certain medications, people with mental health conditions
Incubation Period: none (develops gradually over time)
Symptoms:
- fewer than 3 bowel movements per week
- hard, dry, or lumpy stools
- straining or pain during bowel movements
- feeling of incomplete evacuation
- bloating and abdominal discomfort
- need for manual assistance to pass stool in severe cases
- persistent abdominal pain in severe cases
- nausea or loss of appetite in severe cases
Progression:
1. slowed movement of stool in colon
2. excess water absorption leading to hard stool
3. difficulty passing stool
4. chronic constipation if untreated
Common Locations: colon (large intestine), rectum
Duration: acute (days to weeks); chronic (3 months or more)
Severity: mild_to_moderate
Complications: hemorrhoids (swollen veins in anus), anal fissures (tears in anus), fecal impaction, rectal prolapse
Home Remedy: increase fiber intake (fruits, vegetables, whole grains), drink plenty of water, regular physical activity, establish a regular toilet routine, respond promptly to bowel urges
Avoid: low-fiber processed foods, dehydration, prolonged sitting or inactivity, ignoring the urge to pass stool, excessive use of stimulant laxatives
When to See a Doctor: constipation lasting more than 2-3 weeks, severe pain or difficulty in daily activities, unexplained weight loss, blood in stool or rectal bleeding
Emergency Signs: black or bloody stools, severe abdominal pain, inability to pass stool or gas (possible obstruction)
Prevention: high-fiber diet (25-30g per day), adequate hydration, regular exercise, healthy bowel habits
Contagious Period: none
Special Notes: diagnosis usually based on symptoms and history; may require tests (blood tests, colonoscopy) if persistent; treatment includes lifestyle changes, laxatives, and addressing underlying causes
===
Disease: Diarrhea
Aliases: dysentery (in infectious cases)
Description: a condition characterized by frequent passage of loose or watery stools, often due to infection, irritation, or dysfunction of the digestive system
Cause: bacterial infections, viral infections (e.g., norovirus, rotavirus), parasitic infections, medications (antibiotics, antacids with magnesium, cancer drugs), food intolerances (e.g., lactose intolerance), digestive diseases (e.g., Crohn's disease, Irritable bowel syndrome), post-surgery digestive changes, anxiety or stress
Transmission: often contagious if caused by infections (via contaminated food, water, or poor hygiene)
Risk Groups: children (especially under 5), older adults, travelers to areas with poor sanitation, people with weakened immune systems, individuals taking certain medications
Incubation Period: varies (hours to days depending on cause, especially infections)
Symptoms:
- loose or watery stools (3 or more times per day)
- abdominal cramps or pain
- urgency to pass stool
- bloating
- fever and chills in infectious cases
- nausea or vomiting in infectious cases
- bloody stools in infectious cases
- dehydration (dry mouth, low urine, weakness) in severe cases
- loss of bowel control in severe cases
Progression:
1. irritation or infection in intestines
2. increased fluid secretion in bowel
3. rapid stool movement
4. frequent loose stools
Common Locations: small intestine, large intestine (colon)
Duration: acute (1-3 days, may last up to a week); chronic (4 weeks or more)
Severity: mild_to_severe
Complications: dehydration (most serious), electrolyte imbalance, malnutrition (in chronic cases), kidney problems
Home Remedy: drink plenty of fluids (water, ORS, soups), eat bland foods (rice, bananas, toast), rest, avoid dairy, fatty, and spicy foods
Avoid: contaminated food or water, caffeine and alcohol, high-fat or spicy foods, unnecessary antibiotic use
When to See a Doctor: diarrhea lasting more than 2-3 days (adults), lasting more than 24 hours (children), signs of dehydration, persistent vomiting, high fever
Emergency Signs: severe dehydration (confusion, very little urine), blood or black stools, severe abdominal pain, inability to keep fluids down
Prevention: proper hand hygiene, safe food and water practices, vaccination (e.g., rotavirus for infants), avoid unsafe food while traveling
Contagious Period: during active infection (especially viral or bacterial causes)
Special Notes: usually self-limiting, but hydration is critical; diagnosis may involve stool tests and blood tests; treatment focuses on fluid replacement and addressing underlying cause
===
Disease: Colorectal Cancer
Aliases: colon cancer, rectal cancer
Description: a type of cancer that develops in the tissues of the colon or rectum (parts of the large intestine), often beginning as abnormal growths called polyps that can become cancerous over time
Cause: genetic mutations in DNA (acquired or inherited), progression from precancerous polyps (adenomas), long-term inflammation of the colon (e.g., Ulcerative colitis, Crohn's disease), lifestyle and environmental factors
Transmission: not contagious
Risk Groups: adults over age 45, people with a family history of colorectal cancer, individuals with genetic syndromes (e.g., Lynch syndrome, FAP), people with chronic inflammatory bowel disease, smokers and heavy alcohol users, obese individuals
Incubation Period: long latency (develops over years from polyps to cancer)
Symptoms:
- change in bowel habits (diarrhea or constipation)
- blood in stool (bright red or dark)
- abdominal pain, cramps, or bloating
- feeling of incomplete bowel emptying
- unexplained weight loss
- fatigue
Progression:
1. formation of benign polyps in colon or rectum
2. genetic mutations accumulate
3. polyps become malignant (cancerous)
4. tumor growth and possible spread (metastasis)
Common Locations: colon (large intestine), rectum
Duration: chronic, progressive (develops over years)
Severity: severe
Complications: intestinal blockage, internal bleeding leading to anemia, metastasis to liver, lungs, or other organs, death if untreated
Home Remedy: none (requires medical treatment)
Avoid: smoking, excessive alcohol consumption, sedentary lifestyle, high-fat, low-fiber diet
When to See a Doctor: persistent change in bowel habits, blood in stool, unexplained weight loss, fatigue or weakness
Emergency Signs: severe abdominal pain (possible obstruction), heavy rectal bleeding, signs of advanced disease
Prevention: regular screening (colonoscopy starting around 45 years), removal of precancerous polyps, healthy diet (high fiber, low processed meat), regular exercise, maintaining healthy weight
Contagious Period: none
Special Notes: early stages may have no symptoms, screening is critical; diagnosis involves colonoscopy, biopsy, and imaging; treatment includes surgery, chemotherapy, radiation, targeted therapy, and immunotherapy; early detection significantly improves survival rates
===
Disease: Anemia
Aliases: iron deficiency anemia, low hemoglobin, low RBC count
Description: a common condition in which your body does not have enough healthy red blood cells or hemoglobin to carry adequate oxygen to tissues, leading to fatigue and other symptoms
Cause: iron deficiency (most common), vitamin deficiencies (vitamin B12, folate), chronic diseases (kidney disease, cancer, inflammatory disorders), blood loss (heavy periods, ulcers, surgery, trauma), bone marrow disorders (e.g., aplastic anemia), genetic conditions (e.g., sickle cell disease, thalassemia)
Transmission: not contagious
Risk Groups: women (especially with heavy menstruation), pregnant individuals, children and adolescents (growth phase), elderly, people with poor diet or malnutrition, patients with chronic diseases
Incubation Period: varies (can develop gradually over weeks to months)
Symptoms:
- fatigue (low energy)
- paleness of skin
- shortness of breath
- dizziness or headaches
- cold hands and feet
- rapid or irregular heartbeat
- brittle nails or hair loss
- unusual cravings (pica)
Progression:
1. nutrient deficiency or blood loss or disease begins
2. reduced red blood cell or hemoglobin production
3. decreased oxygen delivery to tissues
4. symptoms like fatigue and weakness appear
Common Locations: blood (red blood cells and hemoglobin)
Duration: can be acute or chronic depending on cause
Severity: mild_to_severe
Complications: heart problems (arrhythmia, heart failure), organ damage due to lack of oxygen, weakened immune system, developmental issues in children, pregnancy complications (low birth weight, premature birth)
Home Remedy: iron-rich diet (spinach, meat, legumes), foods rich in vitamin B12 and folate, vitamin C intake to improve iron absorption
Avoid: poor diet lacking nutrients, excessive tea or coffee (reduces iron absorption), untreated chronic diseases
When to See a Doctor: persistent fatigue or weakness, shortness of breath, dizziness or pale skin
Emergency Signs: chest pain, severe weakness or fainting, rapid or irregular heartbeat
Prevention: balanced diet rich in iron, B12, and folate, regular health check-ups, managing underlying conditions, supplements if recommended by a doctor
Contagious Period: none
Special Notes: diagnosis is usually done via blood tests (CBC); treatment depends on cause (diet, supplements, medicines, transfusion); most cases, especially iron deficiency anemia, are treatable and reversible
===
Disease: Jaundice
Aliases: hyperbilirubinemia, yellowing of skin and eyes
Description: a condition characterized by yellow discoloration of the skin, sclera (whites of the eyes), and mucous membranes due to elevated levels of bilirubin in the blood, often indicating an underlying liver, blood, or bile duct disorder
Cause: excess bilirubin due to breakdown of red blood cells, liver diseases (e.g., Hepatitis, Cirrhosis), blockage of bile ducts (e.g., Gallstones), hemolytic conditions (e.g., Hemolytic anemia), genetic syndromes (e.g., Gilbert syndrome), infections or certain medications
Transmission: not contagious
Risk Groups: newborn babies (physiological jaundice), people with liver disease, individuals with blood disorders, heavy alcohol users, people with bile duct obstruction
Incubation Period: none (depends on underlying cause)
Symptoms:
- yellowing of skin and eyes
- dark urine
- pale or clay-colored stools
- fatigue
- abdominal pain
- itching (in some cases)
Progression:
1. breakdown of red blood cells produces bilirubin
2. liver processes bilirubin and excretes via bile
3. impairment (liver damage or blockage) leads to buildup
4. bilirubin accumulates in blood leading to yellow discoloration
Common Locations: skin, eyes (sclera), blood (bilirubin levels)
Duration: temporary (e.g., newborn jaundice) or chronic depending on cause
Severity: mild_to_severe
Complications: liver failure (in severe liver disease), brain damage in newborns (kernicterus, rare but serious), chronic liver damage
Home Remedy: none specific (focus is on treating underlying cause), maintain hydration and balanced diet
Avoid: alcohol (especially in liver-related jaundice), hepatotoxic drugs
When to See a Doctor: any yellowing of skin or eyes, dark urine or pale stools, persistent fatigue or abdominal pain
Emergency Signs: confusion or altered mental state, severe abdominal pain, rapid worsening of jaundice
Prevention: vaccination for hepatitis (A and B), limit alcohol intake, maintain liver health, avoid unsafe medications or toxins
Contagious Period: none
Special Notes: jaundice is not a disease itself but a sign of an underlying condition; diagnosis often involves a bilirubin blood test and liver function tests; treatment depends entirely on the underlying cause
===
Disease: Infertility
Aliases: subfertility, inability to conceive, fertility problems
Description: a condition in which a couple is unable to achieve pregnancy despite having regular, unprotected sexual intercourse for at least 12 months (or 6 months in women over 35), due to issues affecting either or both partners
Cause: ovulation disorders (e.g., PCOS, hormonal imbalance), sperm abnormalities (low count, poor motility, abnormal shape), blocked fallopian tubes, uterine abnormalities (fibroids, polyps), endometriosis, infections (e.g., chlamydia, gonorrhea), genetic conditions, lifestyle factors (smoking, alcohol, obesity, stress), age-related decline in fertility, sometimes unknown (idiopathic infertility)
Transmission: not contagious
Risk Groups: women over age 35, men over age 40, people with hormonal disorders, individuals with reproductive system diseases, smokers, alcohol users, overweight or underweight individuals
Incubation Period: none
Symptoms:
- inability to conceive after 12 months of trying
- irregular or absent menstrual periods (may vary)
- painful periods (may vary)
- hormonal symptoms (hair growth changes, sexual dysfunction in men) (may vary)
Progression:
1. disruption in ovulation, fertilization, or implantation
2. repeated unsuccessful attempts to conceive
3. possible emotional stress and psychological impact
Common Locations: reproductive organs (ovaries, uterus, fallopian tubes, testes)
Duration: chronic (may persist until treated or resolved)
Severity: mild_to_severe
Complications: psychological stress, anxiety, depression, relationship strain, underlying untreated health conditions
Home Remedy: maintain healthy weight, balanced diet and regular exercise, reduce stress (yoga, meditation), track ovulation cycles
Avoid: smoking and alcohol, excessive caffeine, exposure to toxins and radiation, extreme dieting or over-exercising
When to See a Doctor: no pregnancy after 1 year (or 6 months if over 35), irregular or absent periods, known reproductive health issues, history of miscarriage
Emergency Signs: none specific (not an emergency condition itself)
Prevention: healthy lifestyle habits, managing chronic diseases, avoiding harmful substances, timely reproductive planning
Contagious Period: none
Special Notes: infertility can affect both men and women; many cases are treatable with options like ovulation induction, In vitro fertilization, ICSI, or surgery; some couples may conceive naturally over time even without treatment
===
Disease: Chronic Kidney Disease
Aliases: chronic renal disease, chronic kidney failure, CKD
Description: a long-term condition in which the kidneys gradually lose their ability to filter waste, balance fluids, and regulate essential body functions, leading to accumulation of toxins in the blood over time
Cause: Diabetes (most common cause), Hypertension, heart disease, kidney infections or inflammation, urinary tract blockages, genetic disorders (e.g., polycystic kidney disease)
Transmission: not contagious
Risk Groups: people with diabetes or high blood pressure, elderly individuals, people with family history of kidney disease, individuals with heart disease, certain ethnic groups (higher risk populations)
Incubation Period: none (develops gradually over years)
Symptoms:
- fatigue (early stage often none)
- weakness (early stage often none)
- loss of appetite (early stage often none)
- swelling in feet, ankles, or hands (edema) in advanced stages
- trouble sleeping in advanced stages
- difficulty concentrating in advanced stages
- nausea in advanced stages
- shortness of breath in advanced stages
- blood in urine in advanced stages
Progression:
1. initial kidney damage (often unnoticed)
2. gradual decline in kidney filtration ability
3. buildup of waste and fluid in body
4. complications develop (anemia, bone disease)
5. end-stage kidney failure requiring dialysis or transplant
Common Locations: kidneys
Duration: chronic (long-term, often lifelong)
Severity: mild_to_severe
Complications: Anemia, weak or brittle bones, high blood pressure worsening, heart disease, fluid overload (edema), kidney failure (end-stage renal disease)
Home Remedy: maintain proper hydration, follow kidney-friendly diet (low salt, controlled protein), regular exercise
Avoid: smoking, excessive alcohol, high salt intake, NSAIDs (e.g., ibuprofen) without medical advice
When to See a Doctor: persistent fatigue or swelling, abnormal urine (blood, foam), known risk factors like diabetes or hypertension
Emergency Signs: confusion or severe weakness, difficulty breathing, very little or no urine output
Prevention: control blood sugar and blood pressure, healthy diet and weight management, regular medical checkups, avoid kidney-damaging drugs
Contagious Period: none
Special Notes: CKD often has no symptoms in early stages, making screening important for high-risk individuals; diagnosed using blood tests (creatinine), urine tests (albumin), and blood pressure monitoring; no cure, but progression can be slowed with proper management; advanced stages may require dialysis or kidney transplant
===
Disease: Urinary Tract Infection
Aliases: urine infection, cystitis (bladder infection), urethritis, kidney infection (pyelonephritis), UTI
Description: an infection that occurs in any part of the urinary system - including the kidneys, ureters, bladder, or urethra - most commonly affecting the bladder and urethra, causing discomfort, pain, and urinary symptoms
Cause: bacterial infection (most commonly E. coli), spread of bacteria from the gastrointestinal tract, sexually transmitted infections (e.g., chlamydia, gonorrhea), urinary tract blockages (e.g., kidney stones), catheter use or recent urinary procedures
Transmission: not typically contagious; bacteria enter through urethra (often from own body)
Risk Groups: women (shorter urethra), sexually active individuals, pregnant women, elderly individuals, people with diabetes or weak immune system, catheter users
Incubation Period: usually short (hours to a few days after bacterial entry)
Symptoms:
- burning sensation during urination
- frequent urge to urinate
- passing small amounts of urine
- cloudy, dark, or strong-smelling urine
- pelvic pain
- back or side pain in severe or upper UTI
- fever and chills in severe or upper UTI
- nausea and vomiting in severe or upper UTI
- confusion (especially in elderly) in severe or upper UTI
Progression:
1. bacteria enter urinary tract
2. multiply in bladder (lower UTI)
3. may spread upward to kidneys (upper UTI)
4. complications if untreated
Common Locations: bladder, urethra, kidneys (in severe cases)
Duration: acute (usually resolves in days with treatment); can be recurrent or chronic in some individuals
Severity: mild_to_severe
Complications: kidney damage, recurrent infections, Sepsis (in severe untreated cases), pregnancy complications (low birth weight, premature birth)
Home Remedy: drink plenty of water, urinate frequently, maintain hygiene
Avoid: holding urine for long periods, dehydration, irritating products (perfumed soaps, sprays), spermicidal contraceptives (if prone to UTIs)
When to See a Doctor: burning or pain during urination, frequent urination with discomfort, blood in urine, symptoms lasting more than a few days
Emergency Signs: high fever with chills, severe back pain, confusion or drowsiness, vomiting or inability to keep fluids down
Prevention: drink plenty of fluids, wipe front to back (for women), urinate after sexual activity, maintain proper hygiene, avoid unnecessary catheter use
Contagious Period: none
Special Notes: UTIs are very common and usually easily treated with antibiotics; early treatment prevents complications; recurrent UTIs may need long-term management or further evaluation
===
Disease: Brucellosis
Aliases: undulant fever, Malta fever, Mediterranean fever, Gibraltar fever, Cyprus fever
Description: a zoonotic bacterial infection transmitted from animals to humans, characterized by recurrent fever, sweating, fatigue, and systemic involvement; often acquired through contact with infected animals or consumption of unpasteurized animal products
Cause: bacteria of genus Brucella (commonly Brucella melitensis); infection in livestock such as cattle, goats, camels, pigs, and dogs
Transmission: direct contact with infected animals, tissues, or placenta; consumption of unpasteurized milk or dairy products; ingestion of contaminated meat; inhalation of bacteria in occupational settings
Risk Groups: farmers, veterinarians, slaughterhouse workers, people consuming unpasteurized dairy, travelers to endemic regions (Middle East, Mediterranean, parts of Africa, Latin America)
Incubation Period: typically 1-4 weeks (can vary from days to months)
Symptoms:
- fever with characteristic rising and falling pattern (undulant fever)
- chills and excessive sweating
- fatigue and weakness
- headache
- muscle and joint pain
- loss of appetite
- weight loss
- abdominal pain (other symptom)
- back pain (other symptom)
- swollen lymph nodes (other symptom)
- recurrent fever episodes in chronic cases
- long-term fatigue in chronic cases
- persistent joint pain in chronic cases
Progression:
1. bacteria enter body (ingestion, contact, or inhalation)
2. spread through bloodstream
3. infect organs (liver, spleen, bone marrow)
4. may become chronic if untreated
Common Locations: bloodstream, liver and spleen, bones and joints, central nervous system (in severe cases)
Duration: acute (weeks to months); chronic (months to years if untreated or relapsing)
Severity: moderate_to_severe
Complications: bone and joint infections, Encephalitis, Meningitis, Infective endocarditis (most serious complication)
Home Remedy: none (requires antibiotic treatment)
Avoid: unpasteurized milk and dairy products, direct contact with infected animals without protection
When to See a Doctor: persistent fever with sweating, joint pain and fatigue, history of animal exposure or raw dairy consumption
Emergency Signs: severe headache or neurological symptoms, chest pain or heart-related symptoms, high persistent fever
Prevention: consume only pasteurized dairy products, wear protective gear when handling animals or meat, proper cooking of meat, control and vaccination of livestock
Contagious Period: not typically spread from person to person
Special Notes: commonly seen in developing regions; relapse can occur even after treatment; requires prolonged antibiotic therapy (often 6 weeks or more)
===
Disease: Leptospirosis
Aliases: Weil disease (severe form), leptospira infection
Description: a bacterial zoonotic infection caused by Leptospira species, transmitted through contact with water or soil contaminated with animal urine; it can range from a mild flu-like illness to a severe, life-threatening condition affecting multiple organs
Cause: bacteria of genus Leptospira; exposure to contaminated fresh water or soil; infection reservoirs include rodents, dogs, cattle, and other animals
Transmission: contact with contaminated water, mud, or soil; bacteria entering through cuts, abrasions, or mucous membranes (eyes, nose, mouth); ingestion of contaminated water; rare human-to-human transmission
Risk Groups: farmers and agricultural workers, veterinarians and animal handlers, sewer and sanitation workers, military personnel, people involved in freshwater recreational activities (swimming, kayaking, rafting), individuals in tropical or warm climates
Incubation Period: typically 2-30 days (average around 10 days)
Symptoms:
- fever
- headache
- muscle pain (especially calves and lower back)
- chills
- nausea, vomiting, diarrhea
- dry cough
- abdominal pain (other symptom)
- red eyes (conjunctival suffusion) (other symptom)
- enlarged liver or spleen (other symptom)
- joint pain (other symptom)
- skin rash (other symptom)
- sore throat (other symptom)
- jaundice (Weil disease severe symptom)
- kidney failure (Weil disease severe symptom)
- bleeding disorders (Weil disease severe symptom)
- respiratory distress (Weil disease severe symptom)
Progression:
1. bacteria enter through skin or mucosa
2. spread via bloodstream (leptospiremia phase)
3. immune phase begins with organ involvement
4. severe cases progress to multi-organ dysfunction (Weil disease)
Common Locations: bloodstream, liver, kidneys, lungs, central nervous system
Duration: acute (1-2 weeks); severe or complicated (several weeks)
Severity: mild_to_severe
Complications: Meningitis, kidney failure, liver failure (jaundice), severe bleeding (hemorrhage), respiratory failure
Home Remedy: none (medical treatment required)
Avoid: swimming or wading in contaminated water, contact with animal urine without protection, drinking untreated water
When to See a Doctor: fever with muscle pain after exposure to floodwater or animals, persistent flu-like symptoms with red eyes, history of travel or exposure to contaminated environments
Emergency Signs: yellowing of skin or eyes (jaundice), decreased urine output, breathing difficulty, bleeding or confusion
Prevention: avoid contaminated water sources, wear protective clothing (boots, gloves), control rodents in living areas, ensure safe drinking water, vaccinate animals where applicable
Contagious Period: not typically contagious between humans
Special Notes: more common in tropical and subtropical regions; outbreaks often occur after floods; early antibiotic treatment significantly reduces severity and complications
===
Disease: Acne
Aliases: acne vulgaris, cystic acne, pimples, zits
Description: common skin condition causing blocked hair follicles that develop into whiteheads, blackheads, and inflamed red bumps or cysts
Cause: overproduction of sebum mixing with dead skin cells and bacteria to block hair follicles; excess sebum production, clogged pores, bacterial growth (Cutibacterium acnes), hormonal changes, inflammation
Transmission: none
Risk Groups: teenagers, younger adults, babies, women experiencing hormonal changes, individuals with a family history of acne, adolescents, young adults, individuals with hormonal imbalance, oily skin types
Incubation Period: none
Symptoms:
- blackheads (small black or yellowish bumps)
- whiteheads (firm white spots)
- papules (small tender red bumps)
- pustules (red bumps with a white or yellow pus tip)
- nodules (large hard painful lumps beneath skin)
- cysts (large pus-filled lumps)
- oily skin and skin sensitivity
- post-inflammatory hyperpigmentation or scarring
Progression:
1. sebaceous glands produce excess sebum
2. sebum mixes with dead skin cells
3. mixture blocks hair follicles to form whiteheads or blackheads
4. trapped bacteria cause inflammation leading to papules, pustules, nodules, or cysts
Common Locations: face, shoulders, trunk, back, chest, arms, legs, buttocks
Duration: often goes away after teenage years but may last into middle age, with occasional flare-ups
Severity: mild_to_severe
Complications: skin infections, scarring, emotional stress, depression, anxiety, low self-esteem, hyperpigmentation, psychological distress, secondary infection
Home Remedy: wash gently with mild soap and lukewarm water twice a day, use fragrance-free water-based noncomedogenic products, remove makeup before bed, shampoo daily, gentle cleansing, non-comedogenic skincare, warm compress
Avoid: aggressively squeezing or picking pimples, tight headbands or hats, touching face, greasy cosmetics, excessive skin washing, leaving makeup on overnight, picking or squeezing lesions, harsh scrubbing, excessive cosmetic use, oily products
When to See a Doctor: self-care does not help after several months, severe acne with cysts or extreme redness occurs, condition worsens, scarring develops, emotional stress occurs, baby's acne persists beyond 3 months
Emergency Signs: none
Prevention: wash after exercising, keep hair clean and off the face, avoid greasy hair products, regular skincare routine, balanced diet, stress management, proper hygiene, avoiding pore-clogging products
Contagious Period: none
Special Notes: hormonal therapies like oral contraceptives can help women, isotretinoin is used for severe cystic acne but causes severe birth defects and requires strict monitoring; hormonal fluctuations play a major role, treatment may include topical or oral medications, early treatment reduces risk of scarring
===
Disease: Eczema
Aliases: dermatitis, atopic dermatitis, dyshidrosis, nummular dermatitis, seborrheic dermatitis
Description: skin swelling condition causing dry, itchy skin and rashes that can worsen with scratching
Cause: unknown, likely a combination of genetic and environmental factors; genetic predisposition, immune system overactivity, skin barrier defects, environmental triggers, allergens
Transmission: none
Risk Groups: babies, children, adults, people with hay fever or asthma, individuals with family history of eczema
Incubation Period: none
Symptoms:
- dry skin
- itchy skin
- red rashes
- swelling and increased redness after scratching
- intense itching (pruritus)
- cracked skin
- red or inflamed patches
- thickened or scaly skin in chronic cases
- oozing or crusting in severe cases
Progression:
1. dry skin develops
2. rashes appear in prone areas
3. scratching causes redness, swelling, and more itching
4. chronic skin changes and flare-ups
Common Locations: face, inside elbows, behind knees, hands, feet, neck, wrists
Duration: long-lasting disease, may get better or worse over time, often improves as children grow older; chronic with periodic flare-ups
Severity: mild_to_moderate
Complications: skin infections, sleep disturbance, scarring, psychological distress
Home Remedy: good skin care, skin creams, wet wrap therapy, moisturizers, avoiding triggers, cool compress
Avoid: irritating soaps, certain fabrics, lotions, stress, food allergens, pollen, animals, scratching, harsh soaps, extreme temperatures
When to See a Doctor: persistent or severe symptoms, infection signs, poor response to treatment
Emergency Signs: widespread infection, severe swelling, high fever
Prevention: avoid skin irritants, manage stress, avoid known allergens, regular moisturizing, trigger avoidance, maintaining skin barrier, proper hygiene
Contagious Period: none
Special Notes: often associated with hay fever and asthma, atopic dermatitis is the most common type; early management reduces severity, relapsing condition
===
Disease: Psoriasis
Aliases: plaque psoriasis, psoriasis vulgaris, scalp psoriasis, guttate psoriasis, inverse psoriasis, erythrodermic psoriasis, pustular psoriasis
Description: skin condition causing itchy or sore patches of thick, red, flaky, crusty skin covered with silvery scales due to rapid skin cell turnover
Cause: immune system problem causing healthy skin cells to be attacked and produced too quickly; genetic predisposition, environmental triggers, stress, infections
Transmission: none
Risk Groups: individuals with family history, people with autoimmune conditions, those with stress or infections, people aged 15-35 or 50-60
Incubation Period: none
Symptoms:
- red or pink patches of skin covered with silvery scales
- dry, cracked skin that may bleed
- itching, burning, or soreness around patches
- thickened or pitted nails
- swollen and stiff joints (psoriatic arthritis)
- patches on scalp, elbows, knees, or lower back
Progression:
1. immune system triggers rapid skin cell turnover
2. skin cells build up too rapidly on surface
3. red, scaly patches develop
4. may flare and remit, with triggers worsening symptoms
Common Locations: scalp, elbows, knees, lower back, nails, joints
Duration: lifelong condition; symptoms fluctuate over time
Severity: mild_to_severe
Complications: psoriatic arthritis, cardiovascular disease, depression, anxiety, metabolic syndrome
Home Remedy: moisturizers, mild soaps, sunlight exposure (limited), stress management, oatmeal baths
Avoid: smoking, excessive alcohol, stress, skin injuries, infections
When to See a Doctor: significant skin involvement, joint pain, symptoms not responding to home care, signs of infection
Emergency Signs: widespread redness covering most of the body (erythrodermic psoriasis), severe pain
Prevention: no definite prevention; managing triggers can reduce flares
Contagious Period: none
Special Notes: treated with topical treatments, light therapy, oral medications, or biologics depending on severity
===
Disease: Scabies
Aliases: none
Description: contagious skin condition caused by microscopic mites that burrow into the skin to live and lay eggs
Cause: microscopic mite Sarcoptes scabiei
Transmission: direct skin-to-skin contact, sexual contact, sharing contaminated clothes, towels, or bed linen
Risk Groups: people in crowded environments like schools, nursing homes, or university halls, older people, young children, individuals with lowered immunity
Incubation Period: 2-3 weeks after initial infection
Symptoms:
- intense itching that is often worse at night
- rash made up of tiny red spots or pimple-like irritations
- short, wavy, silver-colored burrow marks with a black dot at one end
- crusty sores from scratching
- thick warty crusts without itching in crusted scabies
Progression:
1. mites transfer to the skin and burrow beneath the surface
2. symptoms of itching and rash appear 2 to 3 weeks later
3. scratching can cause crusty sores and secondary infections
4. immune system reacts to dead mites and droppings, causing itchiness to persist post-treatment
Common Locations: folds of skin between fingers and toes, wrists, underarms, waist, groin, bottom, palms of hands
Duration: requires treatment to cure, post-treatment itchiness can last up to 6 weeks
Severity: mild_to_severe
Complications: secondary skin infection, crusted scabies
Home Remedy: wash all bedding and clothing at 60C or higher, seal un-washable clothing in a bag for 3 days, vacuum carpets and furniture
Avoid: close physical contact, sexual contact, sharing bedding, clothing, or towels until treatment is complete
When to See a Doctor: child under 2 years old has symptoms, skin looks infected, presence of other skin conditions like eczema, crusted or flaky rash appears, itching persists 2 to 4 weeks after pharmacy treatment
Emergency Signs: none
Prevention: treat everyone in the household and sexual partners simultaneously even without symptoms, maintain hygiene
Contagious Period: highly contagious from initial infestation until treatment is completed
Special Notes: cannot be caught from pets as they carry a different mite, permethrin cream and malathion lotion are common treatments
===
Disease: Vitiligo
Aliases: none
Description: skin condition that causes the destruction of color-producing cells, leading to white patches on the skin, eyes, mouth, and nose
Cause: destruction of cells that give skin its color, potentially linked to autoimmune diseases or genetics
Transmission: none
Risk Groups: people with autoimmune diseases, individuals with a family history of vitiligo
Incubation Period: none
Symptoms:
- white patches on skin, especially in sun-exposed areas
- loss of color inside the mouth
- early graying of hair
Progression:
1. color-producing cells in the skin are destroyed
2. white patches begin to appear
3. patches may spread over time to different areas
Common Locations: sun-exposed skin, face, hands, eyes, mouth, nose, hair
Duration: lifelong, patches may spread over time
Severity: mild_to_moderate
Complications: psychological distress, reduced self-esteem
Home Remedy: use sunscreen to protect vulnerable skin, use cosmetics to cover patches
Avoid: unprotected sun exposure
When to See a Doctor: if patches are spreading, for treatment options, for psychological support
Emergency Signs: none
Prevention: none
Contagious Period: none
Special Notes: usually starts before age 40, treatments include medicines, light therapy, and surgery (micropigmentation), but results vary and treatments can have side effects
===
Disease: Hives
Aliases: urticaria
Description: condition characterized by raised, very itchy red bumps or rashes on the skin that can change appearance rapidly
Cause: release of histamine due to allergic reactions, temperature changes, stress, infections, medications, or environmental factors
Transmission: none
Risk Groups: people with other allergies
Incubation Period: none
Symptoms:
- raised red rash or bumps varying in size
- intense itchiness
- patches that spread or change appearance within 24 hours
- swelling of face or lips in some cases
Progression:
1. trigger causes body to release histamine and other chemicals
2. tissues under the skin swell
3. raised itchy rash appears and may spread or change size
Common Locations: anywhere on the body
Duration: usually resolves within a few days, but chronic cases can persist longer
Severity: mild_to_severe
Complications: angioedema, stress, anxiety, anaphylaxis
Home Remedy: antihistamines, menthol cream
Avoid: known triggers like specific foods, insect bites, cold or heat exposure, tight clothing, NSAIDs, alcohol, caffeine
When to See a Doctor: symptoms do not improve after 2 days, rash spreads, hives keep returning, high temperature, swelling under skin, distress
Emergency Signs: swelling in mouth, eyes, face, lips, tongue, or throat, wheezing, lightheadedness, tightness in chest, trouble breathing, abdominal pain, nausea, vomiting
Prevention: identify and avoid personal triggers
Contagious Period: none
Special Notes: in severe cases, doctors may prescribe steroid tablets, and chronic hives may require specialist testing
===
Disease: Skin Cancer
Aliases: melanoma, non-melanoma, basal cell carcinoma, BCC, squamous cell carcinoma, SCC
Description: most common form of cancer involving abnormal cell growth in the skin layers, generally categorized into melanoma and non-melanoma types
Cause: abnormal growth of skin cells such as melanocytes, basal cells, or squamous cells; UV radiation exposure, genetic predisposition
Transmission: none
Risk Groups: people who spend a lot of time in the sun or have been sunburned, individuals with light-colored skin, hair, and eyes, people over age 50, those with a family history of skin cancer
Incubation Period: none
Symptoms:
- suspicious skin markings or moles that change
- changes in the way the skin looks
- new growths that don't heal
- skin lesions that bleed or crust
Progression:
1. abnormal cells develop in the skin
2. suspicious markings or visual changes appear on the skin
3. if untreated, some types of skin cancer cells can spread to other tissues and organs
Common Locations: head, face, neck, hands, arms
Duration: long-term, requires medical treatment
Severity: moderate_to_severe
Complications: spreading of cancer cells to other tissues and organs, metastasis, death
Home Remedy: none
Avoid: excessive sun exposure, tanning beds
When to See a Doctor: notice any suspicious skin markings or changes in the way your skin looks
Emergency Signs: rapidly changing lesion, bleeding that won't stop
Prevention: avoid excessive sun exposure, use sunscreen to protect skin
Contagious Period: none
Special Notes: melanoma is less common but more dangerous, treatments include surgery, radiation therapy, chemotherapy, photodynamic therapy (PDT), and biologic therapy
===
Disease: Boils
Aliases: carbunculosis, folliculitis
Description: a common skin infection affecting groups of hair follicles and nearby tissue that develops into a painful, pus-filled bump
Cause: Staphylococcus aureus bacteria, or other bacteria and fungi entering a damaged hair follicle
Transmission: direct contact with fluid from an infected boil, sharing contaminated washcloths, towels, or bedding
Risk Groups: individuals with diabetes, poor immunity, iron deficiency, skin damage or eczema
Incubation Period: none
Symptoms:
- tender, pinkish-red, swollen spot on firm skin
- pea-sized to golf ball-sized bump that feels like a water-filled balloon
- white or yellow center (pustules)
- weeping, oozing, or crusting
- itching before the boil develops
- skin redness around the boil
- fatigue, fever, general ill-feeling in severe cases
Progression:
1. begins as a tender, pinkish-red, and swollen spot
2. pain worsens as it fills with pus and dead tissue
3. white or yellow center forms
4. boil opens and drains, relieving pain and initiating healing (usually within 2 weeks)
Common Locations: face, neck, armpit, buttocks, thighs, ear canal, nose, spine
Duration: usually opens, drains, and heals within 2 weeks
Severity: mild_to_severe
Complications: carbunculosis (merged boils), abscess of skin or organs, brain infection, heart infection, bone infection, sepsis, permanent scarring
Home Remedy: warm moist compresses several times a day to speed draining, frequent cleaning and dressing changes, thorough hand washing
Avoid: squeezing or cutting the boil open at home, sharing or reusing unwashed towels, washcloths, or clothing
When to See a Doctor: fever, pain or discomfort, boil located on the spine or middle of the face, lasts longer than 1 week, keeps coming back, red streaks, large fluid build-up
Emergency Signs: rapidly spreading redness, fever with severe pain, signs of sepsis
Prevention: thorough hand washing, use of antibacterial soaps and antiseptic washes, washing infected clothing and bedding in hot water
Contagious Period: infectious while draining pus and fluid
Special Notes: antibacterial soaps cannot help much once a boil has already formed
===
Disease: Abdominal Aortic Aneurysm
Aliases: AAA, aortic aneurysm (abdominal)
Description: localized dilation and weakening of the abdominal aorta leading to risk of rupture and life-threatening internal bleeding
Cause: degeneration of aortic wall due to atherosclerosis, hypertension, connective tissue disorders, and aging
Transmission: none
Risk Groups: males over 65, smokers, individuals with hypertension, family history, atherosclerosis patients
Incubation Period: none
Symptoms:
- often asymptomatic in early stages
- deep, constant abdominal or back pain
- pulsating sensation near navel
- sudden severe pain indicating rupture
- dizziness, low blood pressure, fainting in rupture cases
Progression:
1. gradual weakening and dilation of aortic wall
2. aneurysm enlarges over time
3. symptoms may develop as size increases
4. rupture may occur leading to internal bleeding and shock
Common Locations: abdominal aorta (below renal arteries), near navel region
Duration: chronic condition progressing over years
Severity: severe
Complications: rupture, internal bleeding, shock, embolism, organ damage, death
Home Remedy: none
Avoid: smoking, uncontrolled blood pressure, heavy lifting, delayed medical evaluation
When to See a Doctor: persistent abdominal or back pain, pulsating abdominal mass, high-risk screening needs
Emergency Signs: sudden severe abdominal or back pain, fainting, rapid heartbeat, low blood pressure, signs of shock
Prevention: smoking cessation, blood pressure control, regular screening in high-risk individuals, healthy diet, exercise
Contagious Period: none
Special Notes: screening via ultrasound recommended for older men with smoking history, rupture has high mortality rate, early detection significantly improves outcomes
===
Disease: Achilles Tendinopathy
Aliases: Achilles tendonitis, Achilles tendinosis
Description: overuse injury causing pain, stiffness, and degeneration of the Achilles tendon connecting calf muscles to the heel
Cause: repetitive stress, overuse, improper footwear, sudden increase in physical activity, tight calf muscles
Transmission: none
Risk Groups: athletes, runners, middle-aged individuals, people with poor footwear, those with limited flexibility
Incubation Period: none
Symptoms:
- pain and stiffness along the Achilles tendon, especially in the morning
- tenderness or swelling near the heel
- pain during or after physical activity
- thickening of the tendon over time
- reduced range of motion in ankle
Progression:
1. mild pain after activity
2. persistent pain during activity
3. tendon thickening and degeneration
4. possible partial or complete tendon rupture
Common Locations: back of ankle, Achilles tendon region above heel
Duration: weeks to months depending on severity and treatment
Severity: mild_to_moderate
Complications: tendon rupture, chronic pain, reduced mobility, impaired athletic performance
Home Remedy: rest, ice application, stretching exercises, proper footwear, gradual return to activity
Avoid: overexertion, sudden increase in activity, improper footwear, ignoring early symptoms
When to See a Doctor: persistent pain, swelling, difficulty walking, no improvement with rest
Emergency Signs: sudden sharp pain, inability to walk, snapping sensation indicating tendon rupture
Prevention: proper warm-up, gradual training increase, supportive footwear, calf strengthening and flexibility exercises
Contagious Period: none
Special Notes: early management prevents chronic degeneration, eccentric strengthening exercises are beneficial, untreated cases may lead to rupture
===
Disease: Acute Cholecystitis
Aliases: gallbladder inflammation, acute gallbladder infection
Description: sudden inflammation of the gallbladder usually due to obstruction of the cystic duct, leading to pain, infection, and possible complications
Cause: gallstones blocking cystic duct, bile stasis, infection, rarely tumors or trauma
Transmission: none
Risk Groups: females, individuals with gallstones, obesity, rapid weight loss, elderly, pregnancy
Incubation Period: none
Symptoms:
- severe right upper abdominal pain, often after fatty meals
- pain radiating to right shoulder or back
- fever and chills
- nausea and vomiting
- tenderness in upper abdomen (Murphy's sign)
Progression:
1. cystic duct obstruction by gallstone
2. bile accumulation and gallbladder distension
3. inflammation and possible infection
4. complications like necrosis or perforation if untreated
Common Locations: right upper abdomen, gallbladder region
Duration: hours to days (acute episode)
Severity: moderate_to_severe
Complications: gallbladder perforation, abscess, sepsis, gangrene, peritonitis
Home Remedy: none
Avoid: fatty foods, delaying medical care, self-medication without diagnosis
When to See a Doctor: severe abdominal pain, fever, vomiting, suspected gallbladder attack
Emergency Signs: intense abdominal pain, high fever, jaundice, confusion, signs of sepsis
Prevention: healthy diet, weight management, gradual weight loss, management of gallstones
Contagious Period: none
Special Notes: requires prompt medical evaluation, often treated with antibiotics and surgical removal (cholecystectomy), ultrasound is primary diagnostic tool
===
Disease: Acute Lymphoblastic Leukaemia
Aliases: ALL, acute lymphocytic leukemia
Description: rapidly progressing cancer of the blood and bone marrow characterized by overproduction of immature lymphoblasts impairing normal blood cell formation
Cause: genetic mutations affecting lymphoid cell development, chromosomal abnormalities, radiation exposure, certain infections, unknown factors
Transmission: none
Risk Groups: children (most common), elderly, individuals with genetic disorders (e.g., Down syndrome), prior chemotherapy or radiation exposure
Incubation Period: none
Symptoms:
- fatigue and weakness due to anemia
- frequent infections due to low white blood cells
- easy bruising or bleeding (low platelets)
- bone or joint pain
- fever and weight loss
- swollen lymph nodes, liver, or spleen
Progression:
1. genetic mutation in bone marrow cells
2. uncontrolled proliferation of lymphoblasts
3. suppression of normal blood cell production
4. spread to blood, lymph nodes, liver, spleen, and central nervous system
Common Locations: bone marrow, blood, lymph nodes, spleen, liver, central nervous system
Duration: rapidly progressive over weeks to months if untreated
Severity: severe
Complications: severe infections, bleeding, organ infiltration, anemia, treatment-related toxicity, death
Home Remedy: none
Avoid: exposure to infections, delaying treatment, unsupervised medication
When to See a Doctor: persistent fatigue, unexplained bruising, recurrent infections, prolonged fever
Emergency Signs: severe bleeding, high fever, breathing difficulty, neurological symptoms, extreme weakness
Prevention: none
Contagious Period: none
Special Notes: highly treatable especially in children with chemotherapy, early diagnosis improves survival rates, requires long-term monitoring for relapse
===
Disease: Acute Myeloid Leukaemia
Aliases: AML, acute myelogenous leukemia, acute non-lymphocytic leukemia
Description: aggressive cancer of the blood and bone marrow characterized by rapid proliferation of abnormal myeloid cells that interfere with normal hematopoiesis
Cause: genetic mutations in myeloid stem cells, chromosomal abnormalities, prior chemotherapy or radiation, exposure to chemicals like benzene, unknown factors
Transmission: none
Risk Groups: adults over 60, males, individuals with prior blood disorders, exposure to radiation or toxic chemicals, genetic syndromes
Incubation Period: none
Symptoms:
- fatigue and weakness due to anemia
- frequent infections from low functional white cells
- easy bruising or bleeding (low platelets)
- fever and night sweats
- bone or joint pain
- swollen gums, lymph nodes, or spleen
Progression:
1. mutation in myeloid precursor cells in bone marrow
2. rapid proliferation of immature blasts
3. suppression of normal blood cell production
4. infiltration into blood and other organs
Common Locations: bone marrow, blood, spleen, liver, gums, central nervous system
Duration: rapidly progresses over weeks to months if untreated
Severity: severe
Complications: severe infections, hemorrhage, organ failure, leukostasis, treatment toxicity, death
Home Remedy: none
Avoid: exposure to infections, delaying treatment, toxic chemical exposure
When to See a Doctor: persistent fatigue, unexplained bruising, frequent infections, prolonged fever
Emergency Signs: severe bleeding, difficulty breathing, neurological symptoms, high fever, confusion
Prevention: none
Contagious Period: none
Special Notes: requires urgent treatment with chemotherapy, targeted therapy, or stem cell transplant, prognosis varies with age and genetic factors
===
Disease: Acute Respiratory Infection
Aliases: ARI, acute respiratory tract infection, respiratory infection
Description: infection of the upper or lower respiratory tract causing inflammation and symptoms affecting breathing, ranging from mild to severe
Cause: viruses (e.g., influenza virus, respiratory syncytial virus), bacteria (e.g., Streptococcus pneumoniae), environmental irritants
Transmission: airborne droplets (coughing, sneezing), direct contact with infected secretions, contaminated surfaces
Risk Groups: children under 5, elderly, smokers, individuals with weakened immunity, people with chronic respiratory diseases
Incubation Period: 1-7 days depending on pathogen
Symptoms:
- cough and sore throat
- runny or blocked nose
- fever and chills
- difficulty breathing or shortness of breath
- fatigue and body aches
Progression:
1. pathogen enters respiratory tract
2. local inflammation and immune response
3. development of upper or lower respiratory symptoms
4. recovery or progression to severe infection (e.g., pneumonia)
Common Locations: nose, throat, sinuses, bronchi, lungs
Duration: 3-14 days depending on severity and cause
Severity: mild_to_severe
Complications: pneumonia, bronchitis, respiratory failure, sepsis, worsening of chronic diseases
Home Remedy: rest, hydration, warm fluids, steam inhalation, over-the-counter symptom relief
Avoid: smoking, exposure to pollutants, close contact with infected individuals, self-medication without guidance
When to See a Doctor: persistent fever, worsening symptoms, breathing difficulty, high-risk individuals
Emergency Signs: severe breathing difficulty, chest pain, bluish lips or face, confusion, inability to stay awake
Prevention: vaccination (influenza), hand hygiene, mask use, avoiding crowded places during outbreaks
Contagious Period: typically during symptomatic phase and sometimes 1-2 days before symptoms
Special Notes: includes a range of conditions from common cold to pneumonia, severity depends on pathogen and host immunity
===
Disease: Addison's Disease
Aliases: primary adrenal insufficiency, chronic adrenal insufficiency
Description: endocrine disorder characterized by inadequate production of cortisol and often aldosterone due to adrenal gland dysfunction
Cause: autoimmune destruction of adrenal cortex, infections (e.g., tuberculosis), genetic disorders, adrenal hemorrhage, metastasis
Transmission: none
Risk Groups: individuals with autoimmune diseases, genetic predisposition, history of adrenal infections, middle-aged adults
Incubation Period: none
Symptoms:
- chronic fatigue and muscle weakness
- weight loss and decreased appetite
- low blood pressure and dizziness
- hyperpigmentation of skin (darkening)
- salt craving
- nausea, vomiting, abdominal pain
Progression:
1. gradual destruction of adrenal cortex
2. declining production of cortisol and aldosterone
3. worsening symptoms over time
4. risk of acute adrenal crisis under stress
Common Locations: adrenal glands (above kidneys), systemic hormonal effects
Duration: lifelong condition requiring management
Severity: moderate_to_severe
Complications: adrenal crisis, severe hypotension, electrolyte imbalance, shock, death if untreated
Home Remedy: none
Avoid: abrupt discontinuation of steroid medication, unmanaged stress, dehydration
When to See a Doctor: persistent fatigue, unexplained weight loss, skin darkening, low blood pressure symptoms
Emergency Signs: severe weakness, confusion, vomiting, low blood pressure, loss of consciousness (adrenal crisis)
Prevention: none
Contagious Period: none
Special Notes: requires lifelong hormone replacement therapy, stress dosing needed during illness or surgery, early diagnosis prevents life-threatening crises
===
Disease: Adenomyosis
Aliases: uterine adenomyosis
Description: condition where endometrial tissue grows into the muscular wall of the uterus causing pain and heavy menstrual bleeding
Cause: invasion of endometrial cells into uterine muscle, hormonal factors (estrogen), uterine inflammation, prior uterine surgery
Transmission: none
Risk Groups: women aged 30-50, multiparous women, history of uterine surgery (e.g., cesarean), prolonged estrogen exposure
Incubation Period: none
Symptoms:
- heavy or prolonged menstrual bleeding
- severe menstrual cramps (dysmenorrhea)
- chronic pelvic pain
- bloating or enlarged uterus
- pain during intercourse
Progression:
1. endometrial tissue infiltrates uterine muscle
2. cyclic bleeding within muscle layer
3. uterine enlargement and inflammation
4. worsening pain and bleeding over time
Common Locations: uterus (myometrium)
Duration: chronic condition until menopause or treatment
Severity: mild_to_moderate
Complications: anemia due to heavy bleeding, chronic pain, reduced quality of life, fertility issues
Home Remedy: heat therapy, rest, mild pain relief methods
Avoid: delaying medical evaluation, unmanaged pain, excessive physical strain during severe symptoms
When to See a Doctor: heavy bleeding, severe menstrual pain, pelvic discomfort affecting daily life
Emergency Signs: severe bleeding leading to dizziness, fainting, signs of anemia
Prevention: none
Contagious Period: none
Special Notes: often confused with endometriosis, diagnosis may require imaging (MRI or ultrasound), treatment includes hormonal therapy or surgery
===
Disease: Alcohol-Related Liver Disease
Aliases: ARLD, alcoholic liver disease
Description: progressive liver damage caused by excessive alcohol consumption ranging from fatty liver to hepatitis and cirrhosis
Cause: chronic alcohol intake leading to liver inflammation, fat accumulation, oxidative stress, and cellular injury
Transmission: none
Risk Groups: heavy alcohol users, long-term drinkers, males, individuals with poor nutrition, genetic susceptibility
Incubation Period: none
Symptoms:
- fatigue and weakness
- loss of appetite and weight loss
- nausea and vomiting
- abdominal pain and swelling
- jaundice (yellowing of skin and eyes)
- confusion in advanced stages (hepatic encephalopathy)
Progression:
1. fatty liver (steatosis) due to alcohol accumulation
2. alcoholic hepatitis with inflammation
3. fibrosis and scarring of liver tissue
4. cirrhosis leading to liver failure
Common Locations: liver
Duration: months to years depending on alcohol exposure and stage
Severity: mild_to_severe
Complications: cirrhosis, liver failure, portal hypertension, ascites, variceal bleeding, hepatic encephalopathy, liver cancer
Home Remedy: alcohol cessation, balanced nutrition, hydration, rest
Avoid: alcohol consumption, hepatotoxic drugs, poor diet, delaying medical care
When to See a Doctor: jaundice, persistent abdominal pain, swelling, unexplained fatigue, history of heavy drinking
Emergency Signs: vomiting blood, severe confusion, abdominal swelling, difficulty breathing, loss of consciousness
Prevention: limit or avoid alcohol, healthy diet, regular medical check-ups, early intervention in alcohol misuse
Contagious Period: none
Special Notes: early stages may be reversible with abstinence, advanced cirrhosis is irreversible, liver transplantation may be required in severe cases
===
Disease: Allergic Rhinitis
Aliases: hay fever, nasal allergy
Description: allergic inflammation of the nasal passages triggered by exposure to allergens causing sneezing, congestion, and irritation
Cause: immune response to allergens such as pollen, dust mites, mold, animal dander leading to histamine release
Transmission: none
Risk Groups: individuals with family history of allergies, asthma patients, children and young adults, exposure to environmental allergens
Incubation Period: none
Symptoms:
- frequent sneezing
- runny or blocked nose
- itchy nose, eyes, or throat
- watery or red eyes
- postnasal drip and cough
Progression:
1. exposure to allergen
2. immune system activation and histamine release
3. onset of nasal and eye symptoms
4. symptoms persist or recur with continued exposure
Common Locations: nose, nasal passages, sinuses, eyes, throat
Duration: seasonal or perennial depending on allergen exposure
Severity: mild_to_moderate
Complications: sinusitis, ear infections, sleep disturbances, reduced quality of life, asthma exacerbation
Home Remedy: avoid allergens, steam inhalation, saline nasal rinses, maintaining clean environment
Avoid: exposure to allergens, dust, smoke, strong odors, outdoor exposure during high pollen levels
When to See a Doctor: persistent symptoms, difficulty breathing, poor response to over-the-counter treatment
Emergency Signs: severe breathing difficulty, swelling of face or throat, signs of anaphylaxis
Prevention: allergen avoidance, air filters, regular cleaning, use of masks during high pollen seasons
Contagious Period: none
Special Notes: not infectious, often associated with asthma and eczema, antihistamines and nasal steroids are common treatments
===
Disease: Allergies
Aliases: hypersensitivity, allergic reaction
Description: immune system overreaction to typically harmless substances (allergens) causing inflammation and various systemic or localized symptoms
Cause: immune response involving IgE antibodies to allergens such as pollen, dust, foods, medications, insect stings
Transmission: none
Risk Groups: individuals with family history of allergies, asthma or eczema patients, children, people with repeated allergen exposure
Incubation Period: none
Symptoms:
- sneezing, runny or blocked nose
- itchy eyes, skin, or throat
- skin rashes or hives
- swelling of lips, face, or eyelids
- breathing difficulty in severe reactions
Progression:
1. initial sensitization to allergen
2. immune system produces IgE antibodies
3. re-exposure triggers histamine release
4. symptoms develop ranging from mild to severe
Common Locations: nose, skin, eyes, respiratory tract, gastrointestinal system
Duration: minutes to chronic depending on exposure and type
Severity: mild_to_severe
Complications: anaphylaxis, asthma exacerbation, chronic sinusitis, dermatitis, airway obstruction
Home Remedy: allergen avoidance, antihistamine use, cold compress for itching, hydration
Avoid: known allergens, self-exposure without precautions, untreated severe reactions
When to See a Doctor: persistent symptoms, unclear triggers, severe reactions, impact on daily life
Emergency Signs: difficulty breathing, throat swelling, rapid drop in blood pressure, loss of consciousness (anaphylaxis)
Prevention: avoid allergens, use protective measures, immunotherapy in selected cases, maintaining clean environment
Contagious Period: none
Special Notes: reactions vary widely from mild to life-threatening, early identification of triggers is crucial, epinephrine is life-saving in anaphylaxis
===
Disease: Alopecia
Aliases: hair loss, alopecia areata (specific type)
Description: condition characterized by partial or complete loss of hair from the scalp or body due to various underlying factors
Cause: autoimmune reactions, genetic factors, hormonal imbalance, stress, nutritional deficiencies, medical conditions
Transmission: none
Risk Groups: individuals with family history, autoimmune disorders, hormonal imbalance, high stress levels, aging population
Incubation Period: none
Symptoms:
- gradual thinning of hair on scalp
- patchy hair loss (round bald spots)
- sudden shedding of hair
- receding hairline or widening part
- loss of hair on other body parts in severe cases
Progression:
1. disruption of normal hair growth cycle
2. increased hair shedding or follicle damage
3. visible thinning or bald patches
4. possible regrowth or progression depending on cause
Common Locations: scalp, beard area, eyebrows, eyelashes, body hair
Duration: variable, temporary or chronic depending on type
Severity: mild_to_moderate
Complications: psychological distress, reduced self-esteem, permanent hair loss in some cases
Home Remedy: balanced diet, stress management, gentle hair care, scalp hygiene
Avoid: harsh hair treatments, tight hairstyles, excessive heat or chemicals, ignoring underlying conditions
When to See a Doctor: sudden or patchy hair loss, excessive shedding, associated medical symptoms
Emergency Signs: none
Prevention: proper nutrition, stress reduction, gentle hair practices, early treatment of underlying conditions
Contagious Period: none
Special Notes: multiple types exist (androgenetic, areata, telogen effluvium), treatment depends on cause, some forms are reversible
===
Disease: Anal Cancer
Aliases: cancer of the anus, anal carcinoma
Description: malignant tumor arising from the tissues of the anus, often associated with human papillomavirus infection and affecting the anal canal
Cause: human papillomavirus (HPV) infection, chronic inflammation, smoking, immunosuppression
Transmission: none
Risk Groups: individuals with HPV infection, multiple sexual partners, men who have sex with men, HIV-positive individuals, smokers
Incubation Period: none
Symptoms:
- rectal bleeding or blood in stool
- pain or pressure in anal area
- itching or discharge from anus
- lump or mass near anus
- changes in bowel habits
Progression:
1. abnormal cell changes in anal lining
2. development of precancerous lesions
3. growth of malignant tumor
4. possible spread to lymph nodes and distant organs
Common Locations: anal canal, perianal region
Duration: progressive over months to years
Severity: severe
Complications: metastasis, obstruction, severe pain, infection, treatment-related side effects
Home Remedy: none
Avoid: smoking, high-risk sexual behaviors, delaying screening or treatment
When to See a Doctor: persistent anal pain, bleeding, lumps, or unusual discharge
Emergency Signs: severe bleeding, intense pain, bowel obstruction, systemic weakness
Prevention: HPV vaccination, safe sexual practices, smoking cessation, regular screening in high-risk groups
Contagious Period: none
Special Notes: early detection improves outcomes, often treated with chemotherapy and radiation, HPV plays a major role in many cases
===
Disease: Anaphylaxis
Aliases: anaphylactic reaction, severe allergic reaction
Description: life-threatening, rapid-onset systemic allergic reaction causing airway constriction, circulatory collapse, and multi-organ involvement
Cause: IgE-mediated hypersensitivity to allergens such as foods, medications, insect stings, latex
Transmission: none
Risk Groups: individuals with known severe allergies, asthma patients, prior anaphylaxis history, exposure to high-risk allergens
Incubation Period: minutes to hours after exposure
Symptoms:
- difficulty breathing or wheezing
- swelling of face, lips, tongue, or throat
- hives, itching, or skin flushing
- rapid drop in blood pressure (hypotension)
- dizziness, fainting, or loss of consciousness
Progression:
1. exposure to allergen
2. rapid immune activation and histamine release
3. systemic vasodilation and airway constriction
4. shock and multi-organ dysfunction if untreated
Common Locations: systemic (skin, respiratory tract, cardiovascular system, gastrointestinal system)
Duration: minutes to hours (requires immediate treatment)
Severity: severe
Complications: anaphylactic shock, respiratory failure, cardiac arrest, death
Home Remedy: none
Avoid: known allergens, unmonitored exposure, delayed emergency treatment
When to See a Doctor: any suspected allergic reaction with systemic symptoms
Emergency Signs: breathing difficulty, throat swelling, hypotension, unconsciousness
Prevention: allergen avoidance, carrying epinephrine auto-injector, medical alert identification, allergy testing
Contagious Period: none
Special Notes: medical emergency requiring immediate epinephrine administration, recurrence can occur (biphasic reaction), prompt treatment is life-saving
===
Disease: Angina
Aliases: angina pectoris, chest pain due to ischemia
Description: clinical syndrome characterized by chest discomfort caused by reduced blood flow to the heart muscle (myocardial ischemia)
Cause: coronary artery disease due to atherosclerosis, coronary artery spasm, reduced oxygen supply to myocardium
Transmission: none
Risk Groups: older adults, smokers, individuals with hypertension, diabetes, high cholesterol, obesity, sedentary lifestyle
Incubation Period: none
Symptoms:
- chest pain or pressure (tightness, squeezing sensation)
- pain radiating to arms, neck, jaw, shoulder, or back
- shortness of breath
- fatigue or weakness
- nausea or sweating
Progression:
1. narrowing of coronary arteries
2. reduced blood flow during exertion or stress
3. onset of chest pain (stable angina)
4. possible progression to unstable angina or myocardial infarction
Common Locations: chest (retrosternal area), may radiate to upper body
Duration: minutes (typically relieved by rest or medication)
Severity: moderate_to_severe
Complications: myocardial infarction (heart attack), arrhythmias, heart failure, sudden cardiac death
Home Remedy: rest, stress reduction, lifestyle modification, prescribed medications adherence
Avoid: physical overexertion, smoking, high-fat diet, unmanaged stress, skipping medications
When to See a Doctor: new or worsening chest pain, pain at rest, reduced response to medication
Emergency Signs: severe or prolonged chest pain, pain not relieved by rest, breathing difficulty, fainting
Prevention: healthy diet, regular exercise, smoking cessation, control of blood pressure, cholesterol, and diabetes
Contagious Period: none
Special Notes: types include stable, unstable, and variant angina, requires medical evaluation to prevent heart attack
===
Disease: Angioedema
Aliases: angioneurotic edema, Quincke's edema
Description: sudden swelling of deeper layers of skin and mucous membranes due to fluid leakage from blood vessels, often associated with allergic or non-allergic mechanisms
Cause: allergic reactions (foods, medications), hereditary C1 esterase inhibitor deficiency, ACE inhibitor drugs, infections, unknown factors
Transmission: none
Risk Groups: individuals with allergies, family history of hereditary angioedema, those taking ACE inhibitors, asthma patients
Incubation Period: minutes to hours after trigger exposure
Symptoms:
- swelling of face, lips, eyelids, or tongue
- swelling of throat causing difficulty breathing
- abdominal pain, nausea, or vomiting (intestinal involvement)
- skin may feel tight or painful without itching
- hoarseness or difficulty swallowing
Progression:
1. exposure to trigger or genetic activation
2. release of inflammatory mediators or bradykinin
3. increased vascular permeability
4. localized swelling that may resolve or worsen
Common Locations: face, lips, eyelids, tongue, throat, hands, feet, gastrointestinal tract
Duration: hours to a few days depending on type
Severity: mild_to_severe
Complications: airway obstruction, severe abdominal symptoms, recurrence, life-threatening respiratory compromise
Home Remedy: cold compress, avoidance of triggers, antihistamines (if allergic type)
Avoid: known triggers, ACE inhibitors (if related), delaying treatment for severe symptoms
When to See a Doctor: recurrent swelling, unknown cause, associated abdominal pain, breathing discomfort
Emergency Signs: throat swelling, difficulty breathing, voice changes, severe abdominal pain
Prevention: avoid triggers, genetic counseling for hereditary cases, appropriate medication management
Contagious Period: none
Special Notes: may be allergic (histamine-mediated) or non-allergic (bradykinin-mediated), hereditary form does not respond to antihistamines, emergency care needed if airway involved
===
Disease: Ankle Sprain
Aliases: twisted ankle, ligament sprain (ankle)
Description: injury to the ligaments of the ankle caused by overstretching or tearing, commonly due to sudden twisting or rolling of the foot
Cause: inversion or eversion injury, sudden movement, uneven surfaces, sports injuries, inadequate warm-up
Transmission: none
Risk Groups: athletes, runners, individuals with previous ankle injuries, people with poor balance, those wearing improper footwear
Incubation Period: none
Symptoms:
- pain around the ankle
- swelling and bruising
- limited range of motion
- instability or difficulty bearing weight
- tenderness on touch
Progression:
1. sudden ligament stretch or tear
2. immediate pain and swelling
3. bruising and reduced mobility
4. gradual healing or chronic instability if untreated
Common Locations: lateral or medial ligaments of ankle joint
Duration: days to weeks depending on severity
Severity: mild_to_moderate
Complications: chronic instability, recurrent sprains, ligament damage, joint stiffness, arthritis
Home Remedy: rest, ice, compression, elevation (RICE), gentle exercises during recovery
Avoid: weight-bearing too early, intense activity, ignoring injury, improper support
When to See a Doctor: severe pain, inability to walk, significant swelling, suspected fracture
Emergency Signs: extreme pain, deformity, loss of sensation, inability to move foot
Prevention: proper footwear, ankle strengthening exercises, warm-up before activity, caution on uneven surfaces
Contagious Period: none
Special Notes: graded from mild (ligament stretch) to severe (complete tear), rehabilitation is important to prevent recurrence
===
Disease: Ankle Avulsion Fracture
Aliases: avulsion fracture of ankle, ligament avulsion injury
Description: fracture where a small fragment of bone is pulled off at the ankle due to ligament or tendon force during injury
Cause: sudden twisting or rolling of ankle, sports injuries, falls, high-impact trauma causing ligament pull on bone
Transmission: none
Risk Groups: athletes, individuals with previous ankle injuries, elderly with weak bones, people engaging in high-impact activities
Incubation Period: none
Symptoms:
- sudden sharp pain in ankle
- swelling and bruising
- difficulty or inability to bear weight
- tenderness at injury site
- reduced range of motion
Progression:
1. traumatic force causes ligament or tendon pull
2. small bone fragment is detached
3. pain, swelling, and inflammation develop
4. healing with immobilization or possible complications if untreated
Common Locations: ankle joint, lateral malleolus, medial malleolus, base of fifth metatarsal
Duration: weeks to months depending on severity and treatment
Severity: moderate
Complications: non-union, chronic pain, joint instability, reduced mobility, arthritis
Home Remedy: rest, ice, compression, elevation (RICE), immobilization support
Avoid: weight-bearing without support, ignoring injury, early return to activity, improper healing
When to See a Doctor: severe pain, swelling, inability to walk, suspected fracture
Emergency Signs: severe deformity, extreme pain, numbness, loss of circulation
Prevention: proper footwear, strengthening exercises, caution during sports, avoiding uneven surfaces
Contagious Period: none
Special Notes: often misdiagnosed as sprain, imaging (X-ray) required for confirmation, treatment may include casting or surgery in severe cases
===
Disease: Ankylosing Spondylitis
Aliases: AS, axial spondyloarthritis, radiographic axial spondyloarthritis, Bechterew disease
Description: chronic inflammatory autoimmune disease primarily affecting the spine and sacroiliac joints leading to pain and progressive stiffness
Cause: autoimmune inflammation associated with HLA-B27 genetic marker, environmental triggers, immune dysregulation
Transmission: none
Risk Groups: young adults (especially males), individuals with HLA-B27 gene, family history, onset typically before age 40
Incubation Period: none
Symptoms:
- chronic lower back pain and stiffness (worse in morning)
- reduced spinal flexibility
- pain improving with exercise but not rest
- fatigue
- inflammation in other joints or eyes (uveitis)
- pain and swelling in hips, knees, shoulders, ribs, feet
- enthesitis causing heel, rib, or elbow pain where tendons attach
- waking during night due to back pain
Progression:
1. inflammation of sacroiliac joints
2. spread of inflammation to spine
3. formation of new bone and fusion of vertebrae
4. reduced mobility and spinal deformity (kyphosis)
Common Locations: spine, sacroiliac joints, hips, shoulders, eyes
Duration: lifelong progressive condition
Severity: moderate_to_severe
Complications: spinal fusion, reduced mobility, osteoporosis, fractures, uveitis, cardiovascular issues
Home Remedy: regular exercise, posture correction, stretching, heat therapy
Avoid: prolonged inactivity, poor posture, smoking, delayed treatment
When to See a Doctor: persistent back pain in young adults, stiffness lasting more than 3 months, reduced mobility
Emergency Signs: sudden severe pain with fracture, neurological deficits, severe eye pain or vision changes
Prevention: none
Contagious Period: none
Special Notes: early diagnosis improves outcomes, biologic therapies can slow progression, physical therapy is essential
===
Disease: Anorexia Nervosa
Aliases: anorexia, eating disorder (restrictive type)
Description: serious psychiatric eating disorder characterized by self-imposed starvation, intense fear of weight gain, and distorted body image
Cause: multifactorial including psychological factors, genetic predisposition, societal pressures, neurobiological influences
Transmission: none
Risk Groups: adolescents, young adults (especially females), individuals with perfectionist traits, family history of eating disorders, athletes
Incubation Period: none
Symptoms:
- extreme weight loss or low body weight
- intense fear of gaining weight
- distorted body image perception
- restrictive eating or food avoidance
- fatigue, dizziness, or fainting
- hair thinning, dry skin, brittle nails
Progression:
1. restrictive eating behaviors begin
2. significant weight loss and malnutrition
3. physical and psychological deterioration
4. severe complications or life-threatening condition if untreated
Common Locations: systemic (affects multiple body systems including brain, heart, bones)
Duration: months to years, often chronic with relapses
Severity: severe
Complications: malnutrition, electrolyte imbalance, cardiac arrhythmias, osteoporosis, infertility, organ failure, death
Home Remedy: none
Avoid: self-starvation, excessive exercise, ignoring symptoms, delaying professional help
When to See a Doctor: significant weight loss, eating restriction, psychological distress, physical weakness
Emergency Signs: severe weakness, fainting, irregular heartbeat, dehydration, confusion
Prevention: early psychological support, healthy body image education, monitoring at-risk individuals
Contagious Period: none
Special Notes: requires multidisciplinary treatment including medical, nutritional, and psychological care, early intervention improves prognosis
===
Disease: Anxiety Disorders in Children and Young People
Aliases: pediatric anxiety disorders, childhood anxiety, adolescent anxiety disorders
Description: group of mental health conditions in children and adolescents characterized by excessive fear, worry, or nervousness interfering with daily functioning
Cause: combination of genetic predisposition, environmental stressors, brain chemistry imbalance, traumatic experiences, parenting or social factors
Transmission: none
Risk Groups: children with family history of anxiety or mental illness, exposure to trauma or stress, chronic illness, academic or social pressure
Incubation Period: none
Symptoms:
- excessive worry or fear disproportionate to situation
- irritability or restlessness
- difficulty concentrating or sleep disturbances
- physical symptoms like headaches, stomach aches, or fatigue
- avoidance of social or school activities
Progression:
1. early signs of fear or worry in specific situations
2. increasing frequency and intensity of anxiety symptoms
3. interference with school, relationships, and daily life
4. potential development of chronic anxiety or other mental disorders
Common Locations: brain (emotional regulation centers), systemic behavioral impact
Duration: months to years, may be episodic or chronic
Severity: mild_to_severe
Complications: depression, academic difficulties, social withdrawal, substance misuse in later life, reduced quality of life
Home Remedy: supportive environment, relaxation techniques, routine establishment, parental reassurance
Avoid: ignoring symptoms, excessive pressure, negative reinforcement, lack of support
When to See a Doctor: persistent anxiety, impact on school or social life, physical symptoms without clear cause
Emergency Signs: panic attacks with severe distress, self-harm thoughts, inability to function
Prevention: early emotional support, healthy coping skills, stable environment, open communication
Contagious Period: none
Special Notes: early intervention with therapy (e.g., CBT) is effective, family involvement is important, may coexist with other disorders like ADHD or depression
===
Disease: Arterial Thrombosis
Aliases: arterial clot, arterial blood clot
Description: formation of a blood clot within an artery leading to reduced or blocked blood flow to vital organs or tissues
Cause: atherosclerosis, endothelial injury, hypercoagulable states, smoking, hypertension, diabetes
Transmission: none
Risk Groups: older adults, smokers, individuals with cardiovascular disease, diabetes, high cholesterol, obesity
Incubation Period: none
Symptoms:
- sudden pain in affected area
- coldness or pale skin distal to blockage
- numbness or tingling
- weakness or loss of function
- absent or reduced pulse
Progression:
1. plaque formation in artery (atherosclerosis)
2. rupture of plaque and clot formation
3. partial or complete arterial blockage
4. tissue ischemia and possible infarction
Common Locations: coronary arteries, cerebral arteries, peripheral arteries (legs)
Duration: acute onset, may persist until treated
Severity: severe
Complications: myocardial infarction, stroke, limb ischemia, tissue necrosis, organ damage, death
Home Remedy: none
Avoid: smoking, sedentary lifestyle, unmanaged chronic diseases, delayed medical care
When to See a Doctor: sudden limb pain, numbness, weakness, chest pain, neurological symptoms
Emergency Signs: severe chest pain, stroke symptoms, loss of limb function, sudden severe pain with pallor
Prevention: healthy lifestyle, blood pressure and cholesterol control, diabetes management, antiplatelet therapy if prescribed
Contagious Period: none
Special Notes: medical emergency requiring prompt treatment, diagnosis may involve imaging and blood tests, treatment includes anticoagulants or surgical intervention
===
Disease: Arthritis
Aliases: joint inflammation, rheumatic disease (general)
Description: group of conditions causing inflammation, pain, stiffness, and reduced mobility in joints
Cause: autoimmune reactions (e.g., rheumatoid arthritis), wear and tear (osteoarthritis), infections, metabolic disorders (e.g., gout)
Transmission: none
Risk Groups: elderly, females, individuals with family history, obesity, joint injuries, autoimmune conditions
Incubation Period: none
Symptoms:
- joint pain and tenderness
- stiffness, especially in the morning
- swelling and redness around joints
- reduced range of motion
- warmth in affected joints
Progression:
1. inflammation or degeneration begins in joint
2. cartilage damage or immune-mediated attack
3. increased pain and stiffness
4. joint deformity or loss of function in advanced stages
Common Locations: knees, hips, hands, spine, shoulders
Duration: chronic, may be lifelong
Severity: mild_to_severe
Complications: joint deformity, disability, chronic pain, reduced mobility, systemic effects in some types
Home Remedy: exercise, weight management, heat or cold therapy, balanced diet
Avoid: excessive joint strain, inactivity, poor posture, untreated inflammation
When to See a Doctor: persistent joint pain, swelling, stiffness, difficulty in movement
Emergency Signs: severe joint pain with fever, inability to move joint, signs of infection
Prevention: maintaining healthy weight, regular exercise, joint protection, early treatment
Contagious Period: none
Special Notes: includes many types (osteoarthritis, rheumatoid arthritis, gout), treatment varies by type, early diagnosis improves outcomes
===
Disease: Asbestosis
Aliases: pulmonary asbestosis, asbestos-related lung fibrosis
Description: chronic lung disease caused by inhalation of asbestos fibers leading to lung scarring and reduced respiratory function
Cause: prolonged inhalation of asbestos fibers causing inflammation and fibrosis in lung tissue
Transmission: none
Risk Groups: construction workers, shipyard workers, miners, individuals exposed to asbestos in occupational settings, long-term exposure cases
Incubation Period: none
Symptoms:
- shortness of breath, especially on exertion
- persistent dry cough
- chest tightness or pain
- fatigue and weakness
- clubbing of fingers in advanced stages
Progression:
1. inhalation of asbestos fibers
2. fibers lodge in lung tissue causing inflammation
3. gradual fibrosis and thickening of lung tissue
4. progressive decline in lung function
Common Locations: lungs (interstitial tissue)
Duration: chronic, develops over years to decades after exposure
Severity: moderate_to_severe
Complications: respiratory failure, pulmonary hypertension, lung cancer, mesothelioma
Home Remedy: none
Avoid: further asbestos exposure, smoking, delaying medical evaluation
When to See a Doctor: persistent cough, breathing difficulty, history of asbestos exposure
Emergency Signs: severe breathing difficulty, cyanosis, respiratory distress
Prevention: avoiding asbestos exposure, use of protective equipment, workplace safety regulations
Contagious Period: none
Special Notes: no cure, management focuses on symptom relief and preventing progression, smoking increases risk of complications
===
Disease: Asthma
Aliases: bronchial asthma, reactive airway disease
Description: chronic inflammatory airway disease characterized by reversible airway obstruction, bronchospasm, and increased mucus production causing breathing difficulty
Cause: airway inflammation triggered by allergens, infections, exercise, cold air, pollution, genetic predisposition
Transmission: none
Risk Groups: children, individuals with allergies, family history of asthma, smokers, exposure to environmental pollutants
Incubation Period: none
Symptoms:
- wheezing (whistling sound while breathing)
- shortness of breath
- chest tightness or pain
- persistent cough (especially at night or early morning)
- difficulty breathing during triggers
Progression:
1. exposure to trigger
2. airway inflammation and bronchoconstriction
3. narrowing of airways with mucus production
4. symptoms worsen or resolve with treatment
Common Locations: lungs, bronchial airways
Duration: chronic condition with episodic flare-ups
Severity: mild_to_severe
Complications: severe asthma attacks, respiratory failure, reduced quality of life, frequent hospitalizations
Home Remedy: avoid triggers, use prescribed inhalers, breathing exercises, maintain clean environment
Avoid: allergens, smoke, cold air, pollution, skipping medications
When to See a Doctor: frequent symptoms, worsening control, limited daily activities, poor response to treatment
Emergency Signs: severe breathlessness, inability to speak, bluish lips or face, no relief from inhaler
Prevention: trigger avoidance, regular medication use, vaccination, monitoring lung function
Contagious Period: none
Special Notes: reversible airway obstruction, proper management allows normal life, requires long-term monitoring
===
Disease: Ataxia
Aliases: cerebellar ataxia, coordination disorder
Description: neurological condition characterized by impaired coordination, balance, and speech due to dysfunction of the cerebellum or related pathways
Cause: genetic disorders, cerebellar degeneration, stroke, multiple sclerosis, alcohol abuse, infections, vitamin deficiencies
Transmission: none
Risk Groups: individuals with family history (genetic forms), elderly, alcohol users, patients with neurological disorders, vitamin deficiency
Incubation Period: none
Symptoms:
- unsteady gait and difficulty walking
- poor coordination of hands and limbs
- slurred speech (dysarthria)
- difficulty with fine motor tasks
- abnormal eye movements (nystagmus)
Progression:
1. damage or dysfunction of cerebellum
2. onset of coordination and balance problems
3. worsening motor control and speech difficulties
4. possible disability depending on cause
Common Locations: cerebellum, brainstem, peripheral nerves
Duration: variable, acute or chronic depending on cause
Severity: mild_to_severe
Complications: falls, injury, disability, difficulty performing daily activities
Home Remedy: physical therapy, balance exercises, supportive care
Avoid: alcohol consumption, unsafe environments, delaying medical evaluation
When to See a Doctor: persistent coordination issues, difficulty walking, speech changes
Emergency Signs: sudden onset ataxia, severe headache, weakness, vision changes (possible stroke)
Prevention: managing underlying conditions, proper nutrition, avoiding toxins
Contagious Period: none
Special Notes: may be hereditary or acquired, treatment focuses on underlying cause and rehabilitation
===
Disease: Atrial Fibrillation
Aliases: AF, AFib
Description: common cardiac arrhythmia characterized by irregular and often rapid heartbeat due to disorganized electrical activity in the atria
Cause: hypertension, coronary artery disease, heart valve disorders, hyperthyroidism, alcohol use, aging
Transmission: none
Risk Groups: elderly, individuals with heart disease, hypertension, diabetes, obesity, excessive alcohol use
Incubation Period: none
Symptoms:
- irregular or rapid heartbeat (palpitations)
- shortness of breath
- fatigue or weakness
- dizziness or lightheadedness
- chest discomfort
Progression:
1. abnormal electrical signals in atria
2. irregular atrial contractions
3. inefficient blood flow and pooling in atria
4. increased risk of clot formation and complications
Common Locations: heart (atria)
Duration: episodic or chronic, may be persistent
Severity: moderate_to_severe
Complications: stroke, heart failure, blood clots, reduced cardiac output
Home Remedy: lifestyle modification, stress management, adherence to prescribed medications
Avoid: excessive alcohol, smoking, unmanaged hypertension, skipping medications
When to See a Doctor: palpitations, irregular heartbeat, shortness of breath, fatigue
Emergency Signs: chest pain, severe breathlessness, fainting, stroke symptoms
Prevention: control of blood pressure, healthy lifestyle, weight management, regular monitoring
Contagious Period: none
Special Notes: may be paroxysmal, persistent, or permanent, anticoagulation often required to prevent stroke, early management reduces complications
===
Disease: Attention Deficit Hyperactivity Disorder
Aliases: ADHD, attention deficit disorder (ADD)
Description: neurodevelopmental disorder characterized by persistent patterns of inattention, hyperactivity, and impulsivity affecting functioning or development
Cause: genetic factors, brain structure and neurotransmitter differences, prenatal exposures, environmental influences
Transmission: none
Risk Groups: children, males, family history, premature birth, low birth weight, exposure to toxins
Incubation Period: none
Symptoms:
- difficulty sustaining attention
- hyperactivity (restlessness, excessive movement)
- impulsivity (interrupting, acting without thinking)
- disorganization and forgetfulness
- difficulty following instructions
Progression:
1. early signs in childhood
2. persistent symptoms affecting school and behavior
3. challenges in adolescence and adulthood
4. possible improvement or persistence over time
Common Locations: brain (prefrontal cortex and related networks)
Duration: chronic, often persists into adulthood
Severity: mild_to_severe
Complications: academic difficulties, social issues, low self-esteem, substance misuse, comorbid mental disorders
Home Remedy: structured routine, behavioral strategies, parental support, lifestyle management
Avoid: inconsistent routines, excessive screen time, unmanaged stress, lack of support
When to See a Doctor: persistent attention or behavior problems, impact on school or daily life
Emergency Signs: none
Prevention: none
Contagious Period: none
Special Notes: multimodal treatment includes behavioral therapy and medication, early intervention improves outcomes, individualized support is important
===
Disease: Autism
Aliases: autism spectrum disorder, ASD
Description: neurodevelopmental disorder characterized by difficulties in social communication, restricted interests, and repetitive behaviors with varying severity
Cause: genetic factors, brain development differences, prenatal influences, environmental factors (not fully understood)
Transmission: none
Risk Groups: children, males, family history, genetic conditions (e.g., fragile X syndrome), premature birth
Incubation Period: none
Symptoms:
- difficulty with social interaction and communication
- delayed speech or language development
- repetitive behaviors or movements
- restricted or intense interests
- sensitivity to sensory stimuli (sound, light, touch)
Progression:
1. early developmental differences in infancy or toddler years
2. noticeable delays in communication and social skills
3. persistent behavioral patterns
4. lifelong condition with varying levels of independence
Common Locations: brain (neurodevelopmental pathways)
Duration: lifelong
Severity: mild_to_severe
Complications: learning difficulties, social challenges, anxiety, behavioral issues, need for support services
Home Remedy: structured environment, behavioral therapy, communication support, parental involvement
Avoid: lack of early intervention, overstimulation, inconsistent routines, ignoring developmental concerns
When to See a Doctor: delayed milestones, lack of eye contact, communication difficulties, behavioral concerns
Emergency Signs: none
Prevention: none
Contagious Period: none
Special Notes: spectrum condition with wide variability, early diagnosis and intervention improve outcomes, individualized therapy plans are essential
===
Disease: Tuberculosis
Aliases: TB, pulmonary tuberculosis, extrapulmonary tuberculosis
Description: infectious bacterial disease primarily affecting the lungs but can involve other organs such as kidneys, spine, and brain
Cause: Mycobacterium tuberculosis bacteria causing infection and tissue damage
Transmission: airborne droplets from coughing, sneezing, talking, or singing by infected individuals
Risk Groups: people in close contact with infected individuals, healthcare workers, immunocompromised individuals, elderly, children, people in crowded living conditions
Incubation Period: weeks to years (latent phase possible before active disease develops)
Symptoms:
- persistent cough lasting more than 3 weeks
- coughing up blood or sputum
- fever and chills
- night sweats
- weight loss and loss of appetite
- fatigue and weakness
- chest pain or breathlessness
Progression:
1. inhalation of TB bacteria
2. latent infection where bacteria remain inactive
3. activation of bacteria when immunity weakens
4. active disease with symptoms and potential spread
Common Locations: lungs (most common), lymph nodes, spine, kidneys, brain
Duration: months to years depending on treatment and disease stage
Severity: moderate_to_severe
Complications: lung damage, meningitis, organ failure, respiratory failure, death if untreated
Home Remedy: none
Avoid: close contact with infected individuals, incomplete antibiotic treatment, poor ventilation environments
When to See a Doctor: persistent cough, weight loss, fever, night sweats, coughing blood
Emergency Signs: severe breathing difficulty, confusion, seizures, coughing significant blood
Prevention: BCG vaccination, good ventilation, covering mouth when coughing, early diagnosis and treatment
Contagious Period: active pulmonary TB until effective treatment reduces infectivity (usually 2-3 weeks after starting treatment)
Special Notes: latent TB is not contagious but can become active, long-term antibiotic treatment required, drug-resistant TB requires extended therapy
===
Disease: COVID-19
Aliases: coronavirus disease 2019, SARS-CoV-2 infection
Description: infectious respiratory illness caused by a novel coronavirus affecting the lungs and multiple body systems with variable severity
Cause: SARS-CoV-2 virus causing infection and systemic inflammatory response
Transmission: airborne droplets and aerosols from infected person, contact with contaminated surfaces, entry via eyes, nose, or mouth
Risk Groups: elderly, immunocompromised individuals, people with chronic diseases, pregnant individuals, individuals with disabilities
Incubation Period: 2-14 days (commonly around 5 days)
Symptoms:
- cough, sore throat, runny or blocked nose
- fever and chills
- fatigue and body aches
- shortness of breath or chest tightness
- headache and loss of appetite
- nausea, vomiting, or diarrhea (less common)
- loss of smell or taste (in some cases)
Progression:
1. exposure to virus via respiratory droplets
2. viral replication in respiratory tract
3. onset of mild to moderate symptoms
4. recovery or progression to severe disease with respiratory complications
Common Locations: lungs, respiratory tract, may affect heart, brain, kidneys, and other organs
Duration: typically 1-3 weeks (longer in severe or long COVID cases)
Severity: mild_to_severe
Complications: pneumonia, acute respiratory distress syndrome, organ failure, long COVID, blood clots, death
Home Remedy: rest, hydration, over-the-counter symptom relief, isolation to prevent spread
Avoid: close contact with others, crowded places, smoking, ignoring worsening symptoms
When to See a Doctor: persistent fever, worsening symptoms, breathing difficulty, high-risk individuals
Emergency Signs: severe breathing difficulty, chest pain, confusion, bluish lips or skin, coughing blood
Prevention: vaccination, mask use, hand hygiene, ventilation, avoiding close contact with infected individuals
Contagious Period: typically from 1-2 days before symptoms until several days after (varies by severity)
Special Notes: can be asymptomatic yet contagious, variants may alter transmissibility and severity, long COVID may occur even after mild illness
===
Disease: Influenza
Aliases: flu, influenza A, influenza B
Description: contagious viral respiratory illness affecting the nose, throat, and lungs with sudden onset of systemic and respiratory symptoms
Cause: influenza viruses (types A and B) infecting the respiratory tract
Transmission: airborne droplets from coughing or sneezing, contact with contaminated surfaces followed by touching face
Risk Groups: elderly, young children, pregnant women, individuals with chronic diseases, immunocompromised individuals
Incubation Period: 1-4 days (commonly 2-3 days)
Symptoms:
- sudden fever and chills
- body aches and headache
- fatigue and weakness
- cough and sore throat
- runny or stuffy nose
- nausea, vomiting, or diarrhea (more common in children)
Progression:
1. exposure to influenza virus
2. rapid viral replication in respiratory tract
3. sudden onset of systemic and respiratory symptoms
4. recovery or complications in high-risk individuals
Common Locations: nose, throat, lungs
Duration: 1-2 weeks (fatigue and cough may last longer)
Severity: mild_to_severe
Complications: pneumonia, bronchitis, sinus infection, ear infection, myocarditis, encephalitis, multi-organ failure
Home Remedy: rest, hydration, over-the-counter medications for fever and pain relief
Avoid: smoking, alcohol, unnecessary antibiotic use, close contact with others when infected
When to See a Doctor: severe symptoms, high-risk individuals, symptoms not improving
Emergency Signs: difficulty breathing, chest pain, persistent high fever, confusion, seizures
Prevention: annual influenza vaccination, hand hygiene, mask use, avoiding close contact with infected individuals
Contagious Period: from about 1 day before symptoms to 5-7 days after onset
Special Notes: antiviral medications may reduce severity if started early, antibiotics are ineffective against viral infection, confusion with cold and COVID-19 is common
===
Disease: Dengue
Aliases: dengue fever, breakbone fever
Description: mosquito-borne viral infection causing fever, rash, and body aches, with potential progression to severe life-threatening disease
Cause: dengue virus (four related serotypes: DENV-1, DENV-2, DENV-3, DENV-4)
Transmission: bite of infected Aedes mosquitoes, rarely via blood transfusion, organ transplant, or from mother to baby during pregnancy
Risk Groups: people living in tropical and subtropical regions, travelers to endemic areas, infants, pregnant women, individuals with prior dengue infection
Incubation Period: 4-10 days after mosquito bite
Symptoms:
- high fever
- severe headache and pain behind the eyes
- muscle, joint, and bone pain
- nausea and vomiting
- skin rash
- mild bleeding (nose or gums)
Progression:
1. infection via mosquito bite
2. viral replication and onset of fever
3. symptomatic phase with systemic symptoms
4. recovery or progression to severe dengue with complications
Common Locations: systemic (blood, liver, vascular system)
Duration: 2-7 days (acute phase)
Severity: mild_to_severe
Complications: severe dengue (dengue hemorrhagic fever), internal bleeding, shock, organ failure, death
Home Remedy: rest, hydration, acetaminophen for fever and pain relief
Avoid: aspirin, ibuprofen, dehydration, mosquito exposure, delayed medical care
When to See a Doctor: high fever, severe pain, vomiting, signs of dehydration, recent travel to endemic areas
Emergency Signs: severe abdominal pain, persistent vomiting, bleeding, blood in stool or vomit, extreme fatigue, restlessness
Prevention: mosquito control, insect repellents, protective clothing, eliminating standing water, vaccination in specific populations
Contagious Period: none (not spread person-to-person)
Special Notes: second infection increases risk of severe dengue, no specific antiviral treatment, early supportive care improves outcomes
===
Disease: Malaria
Aliases: none
Description: serious mosquito-borne parasitic disease causing fever and systemic illness, potentially life-threatening if untreated
Cause: Plasmodium parasites (e.g., Plasmodium falciparum, P. vivax, P. ovale, P. malariae)
Transmission: bite of infected female Anopheles mosquitoes, rarely via blood transfusion or from mother to fetus
Risk Groups: people in tropical and subtropical regions, travelers to endemic areas, children, pregnant women, immunocompromised individuals
Incubation Period: 7-30 days depending on parasite type
Symptoms:
- fever and chills
- flu-like symptoms
- headache and muscle aches
- nausea, vomiting, or diarrhea
- sweating
- jaundice in severe cases
Progression:
1. mosquito bite introduces parasites into bloodstream
2. parasites infect liver cells and multiply
3. release into bloodstream infecting red blood cells
4. cyclical destruction of red cells causing fever and complications
Common Locations: blood, liver, spleen
Duration: days to weeks depending on treatment and severity
Severity: moderate_to_severe
Complications: severe anemia, cerebral malaria, organ failure, hypoglycemia, death
Home Remedy: none
Avoid: mosquito exposure, delayed treatment, travel without prophylaxis
When to See a Doctor: fever after travel to endemic area, chills, flu-like symptoms
Emergency Signs: confusion, seizures, severe anemia, difficulty breathing, jaundice, loss of consciousness
Prevention: antimalarial prophylaxis, insect repellent (DEET), mosquito nets, protective clothing
Contagious Period: none (not spread person-to-person)
Special Notes: most deadly form is Plasmodium falciparum, early diagnosis and treatment are critical, resistance to drugs can occur
===
Disease: Chikungunya
Aliases: chikungunya fever
Description: mosquito-borne viral infection causing sudden fever and severe joint pain, often leading to prolonged joint symptoms
Cause: chikungunya virus (an alphavirus)
Transmission: bite of infected Aedes mosquitoes, rarely from mother to newborn during birth, or via infected blood exposure
Risk Groups: people in endemic regions, travelers, newborns, elderly, individuals with chronic diseases (e.g., diabetes, hypertension, heart disease)
Incubation Period: 3-7 days after mosquito bite
Symptoms:
- sudden high fever
- severe joint pain (often in hands and feet)
- headache
- muscle pain
- joint swelling
- skin rash
Progression:
1. infection via mosquito bite
2. viral replication and onset of fever
3. development of joint pain and systemic symptoms
4. recovery or prolonged joint pain lasting months in some cases
Common Locations: joints, muscles, blood
Duration: 1-2 weeks (joint pain may persist for months)
Severity: mild_to_moderate
Complications: chronic joint pain, neurological complications (rare), organ involvement (rare)
Home Remedy: rest, hydration, non-aspirin pain relief, supportive care
Avoid: mosquito exposure, aspirin (risk of bleeding), dehydration
When to See a Doctor: persistent fever, severe joint pain, recent travel to endemic areas
Emergency Signs: severe weakness, neurological symptoms, signs of organ involvement
Prevention: insect repellents, protective clothing, mosquito control, vaccination in high-risk adults
Contagious Period: none (not spread person-to-person typically)
Special Notes: similar symptoms to dengue and Zika, joint pain can be long-lasting, no specific antiviral treatment available
===
Disease: Typhoid Fever
Aliases: enteric fever
Description: bacterial infection causing prolonged fever, gastrointestinal symptoms, and systemic illness due to bloodstream spread
Cause: Salmonella typhi bacteria infecting intestines and bloodstream
Transmission: ingestion of contaminated food or water, poor sanitation, carriers shedding bacteria in stool
Risk Groups: people in developing regions, travelers to endemic areas, individuals with poor hygiene or sanitation, close contacts of infected carriers
Incubation Period: 6-30 days
Symptoms:
- prolonged high fever
- abdominal pain
- diarrhea or constipation
- weakness and fatigue
- rash (rose spots on abdomen and chest)
- headache and confusion
Progression:
1. ingestion of contaminated food or water
2. bacteria invade intestines and enter bloodstream
3. spread to organs such as liver, spleen, and gallbladder
4. worsening systemic symptoms and possible complications
Common Locations: intestines, blood, liver, spleen, gallbladder
Duration: weeks (typically 2-4 weeks with treatment)
Severity: moderate_to_severe
Complications: intestinal bleeding, intestinal perforation, peritonitis, kidney failure, relapse
Home Remedy: hydration, rest, supportive care
Avoid: contaminated food or water, poor hygiene, incomplete antibiotic treatment
When to See a Doctor: persistent fever, abdominal pain, recent travel to endemic area, gastrointestinal symptoms
Emergency Signs: severe abdominal pain, bleeding, confusion, reduced urine output, signs of perforation
Prevention: vaccination, safe drinking water, proper sanitation, hand hygiene, safe food practices
Contagious Period: while bacteria are present in stool (including carrier state)
Special Notes: carriers can spread infection for years, antibiotic resistance is increasing, early treatment improves outcomes
===
Disease: Cholera
Aliases: none
Description: acute diarrheal bacterial infection causing rapid fluid loss and dehydration, potentially life-threatening if untreated
Cause: Vibrio cholerae bacteria transmitted via contaminated food or water
Transmission: ingestion of contaminated water or food, poor sanitation, fecal contamination (rarely direct person-to-person)
Risk Groups: people in areas with poor sanitation, travelers to endemic regions, disaster-affected populations, individuals with limited access to clean water
Incubation Period: 2-3 days
Symptoms:
- watery diarrhea (rice-water stools)
- vomiting
- leg cramps
- dehydration
- weakness and rapid fluid loss
Progression:
1. ingestion of contaminated food or water
2. bacteria colonize small intestine
3. toxin production leads to fluid secretion
4. rapid dehydration and potential shock if untreated
Common Locations: intestines (small intestine)
Duration: days (can progress rapidly within hours in severe cases)
Severity: mild_to_severe
Complications: severe dehydration, electrolyte imbalance, shock, kidney failure, death
Home Remedy: oral rehydration solution, fluids, rest
Avoid: contaminated water or food, poor hygiene, delayed treatment
When to See a Doctor: severe diarrhea, signs of dehydration, recent travel to affected areas
Emergency Signs: extreme dehydration, confusion, low blood pressure, minimal urine output, rapid pulse
Prevention: clean drinking water, proper sanitation, hand hygiene, safe food practices, vaccination in high-risk areas
Contagious Period: while bacteria are present in stool
Special Notes: rapid treatment with rehydration is highly effective, severe cases may require IV fluids and antibiotics, outbreaks often linked to disasters or poor sanitation
===
Disease: Hepatitis A
Aliases: HAV infection, infectious hepatitis
Description: acute viral infection causing liver inflammation and temporary liver dysfunction, usually self-limiting
Cause: hepatitis A virus (HAV) infecting liver cells
Transmission: ingestion of contaminated food or water, contact with infected stool, close personal contact
Risk Groups: travelers to endemic regions, people with poor sanitation, drug users, men who have sex with men, close contacts of infected individuals
Incubation Period: 2-7 weeks
Symptoms:
- fatigue and weakness
- fever
- nausea and vomiting
- abdominal pain
- loss of appetite
- dark urine and pale stools
- jaundice (yellowing of skin and eyes)
Progression:
1. ingestion of virus through contaminated sources
2. viral replication in liver
3. onset of symptoms and liver inflammation
4. recovery without chronic infection
Common Locations: liver
Duration: weeks to months (usually resolves within 6 months)
Severity: mild_to_moderate
Complications: liver failure (rare), prolonged symptoms, higher risk in older adults or those with liver disease
Home Remedy: rest, hydration, healthy diet, avoid alcohol
Avoid: alcohol, hepatotoxic drugs, poor hygiene, contaminated food or water
When to See a Doctor: jaundice, persistent vomiting, abdominal pain, suspected exposure
Emergency Signs: confusion, severe vomiting, signs of liver failure, altered consciousness
Prevention: hepatitis A vaccination, hand hygiene, safe food and water practices
Contagious Period: about 2 weeks before symptoms to up to 3 weeks after symptom onset
Special Notes: does not cause chronic liver disease, immunity develops after infection or vaccination, highly preventable
===
Disease: Hepatitis B
Aliases: HBV infection, hep B
Description: viral liver infection that can be acute or chronic, causing liver inflammation and potential long-term liver damage
Cause: hepatitis B virus (HBV) infecting liver cells
Transmission: contact with infected blood or body fluids, unprotected sexual contact, sharing needles, mother-to-child during childbirth
Risk Groups: healthcare workers, people with multiple sexual partners, intravenous drug users, travelers to endemic regions, infants born to infected mothers
Incubation Period: 1-4 months
Symptoms:
- loss of appetite
- nausea and vomiting
- fatigue and weakness
- abdominal pain (liver area)
- dark urine
- jaundice (yellowing of skin and eyes)
- joint pain
Progression:
1. exposure to infected blood or body fluids
2. viral replication in liver
3. acute infection with symptoms or asymptomatic phase
4. recovery or progression to chronic infection and liver damage
Common Locations: liver
Duration: weeks to lifelong (chronic cases)
Severity: mild_to_severe
Complications: chronic hepatitis, cirrhosis, liver failure, liver cancer, hepatitis D co-infection
Home Remedy: rest, hydration, healthy diet, avoid alcohol
Avoid: alcohol, sharing needles, unprotected sex, hepatotoxic substances
When to See a Doctor: jaundice, persistent fatigue, abdominal pain, exposure risk
Emergency Signs: severe abdominal pain, confusion, signs of liver failure, vomiting blood
Prevention: hepatitis B vaccination, safe sex practices, sterile needle use, screening of blood products
Contagious Period: while virus is present in blood and body fluids (can be lifelong in chronic cases)
Special Notes: chronic infection risk higher in infants, antiviral therapy may be required, lifelong monitoring needed in chronic cases
===
Disease: Hepatitis C
Aliases: HCV infection, hep C
Description: viral liver infection that can be acute or chronic, often leading to long-term liver damage if untreated
Cause: hepatitis C virus (HCV) infecting liver cells
Transmission: contact with infected blood (sharing needles, unsafe medical procedures, transfusions before screening, unprotected sex, mother-to-child)
Risk Groups: intravenous drug users, healthcare workers, people with multiple sexual partners, individuals with tattoos or piercings using unsterile equipment, people with HIV, dialysis patients
Incubation Period: 2 weeks to 6 months
Symptoms:
- fatigue
- fever
- nausea and vomiting
- abdominal pain
- loss of appetite
- dark urine and pale stools
- jaundice (yellowing of skin and eyes)
- joint pain
Progression:
1. exposure to infected blood
2. viral replication in liver
3. acute infection (often asymptomatic)
4. progression to chronic infection and long-term liver damage
Common Locations: liver
Duration: weeks (acute) to lifelong (chronic)
Severity: mild_to_severe
Complications: cirrhosis, liver failure, liver cancer, chronic liver disease
Home Remedy: rest, hydration, healthy diet, avoid alcohol
Avoid: sharing needles, unprotected sex, alcohol, unsafe medical or cosmetic procedures
When to See a Doctor: risk exposure, abnormal liver tests, fatigue, jaundice
Emergency Signs: confusion, severe abdominal swelling, vomiting blood, signs of liver failure
Prevention: safe injection practices, screened blood supply, protective measures during contact with blood, safe sex practices
Contagious Period: while virus is present in blood (often chronic)
Special Notes: often asymptomatic for years, no vaccine available, antiviral treatment can cure most cases if detected early
===
Disease: Hepatitis E
Aliases: HEV infection
Description: viral infection causing liver inflammation, usually acute and self-limiting but potentially severe in some cases
Cause: hepatitis E virus (HEV) infecting liver cells
Transmission: ingestion of contaminated water, undercooked pork or wild meat, contact with infected animal stool, rarely via blood transfusion
Risk Groups: people in areas with poor sanitation, travelers to endemic regions, pregnant women, immunocompromised individuals, older adults
Incubation Period: 15-60 days
Symptoms:
- fatigue
- fever
- nausea and vomiting
- abdominal pain
- loss of appetite
- dark urine and pale stools
- jaundice (yellowing of skin and eyes)
- joint pain
Progression:
1. ingestion of virus via contaminated sources
2. viral replication in liver
3. onset of symptoms and liver inflammation
4. recovery or rare progression to severe liver failure
Common Locations: liver
Duration: weeks (typically resolves within a few weeks)
Severity: mild_to_severe
Complications: acute liver failure (especially in pregnant women), chronic infection (rare), cirrhosis in immunocompromised patients
Home Remedy: rest, hydration, healthy diet, avoid alcohol
Avoid: contaminated water, undercooked meat, alcohol, poor hygiene
When to See a Doctor: jaundice, persistent vomiting, abdominal pain, travel exposure
Emergency Signs: confusion, severe weakness, signs of liver failure, altered consciousness
Prevention: safe drinking water, proper food cooking, hygiene practices, avoiding contaminated sources
Contagious Period: not commonly spread person-to-person, mainly during active infection
Special Notes: most common viral hepatitis globally, no widely available vaccine in many countries, higher risk in pregnancy
===
Disease: HIV/AIDS
Aliases: human immunodeficiency virus infection, acquired immunodeficiency syndrome
Description: chronic viral infection that attacks the immune system (CD4 cells) leading to progressive immune failure and opportunistic infections
Cause: human immunodeficiency virus (HIV) destroying immune cells and impairing immune response
Transmission: unprotected sexual contact, sharing contaminated needles, blood transfusion (unscreened), mother-to-child during pregnancy, childbirth, or breastfeeding
Risk Groups: individuals with multiple sexual partners, intravenous drug users, healthcare exposure, infants born to infected mothers, people with unprotected sex
Incubation Period: 2-4 weeks (acute symptoms), progression to AIDS may take years
Symptoms:
- fever and sore throat (early stage)
- swollen lymph nodes
- fatigue and weight loss
- recurrent infections
- night sweats
- chronic diarrhea
Progression:
1. acute HIV infection with flu-like symptoms
2. asymptomatic or latent phase with viral replication
3. gradual immune system decline
4. advanced stage (AIDS) with opportunistic infections and severe illness
Common Locations: immune system (CD4 T cells), blood, lymph nodes
Duration: lifelong without cure
Severity: severe
Complications: opportunistic infections, cancers (e.g., Kaposi sarcoma), neurological disorders, wasting syndrome, death
Home Remedy: none
Avoid: unprotected sex, sharing needles, exposure to infected blood, delayed treatment
When to See a Doctor: risk exposure, persistent fever, weight loss, recurrent infections
Emergency Signs: severe infections, neurological symptoms, extreme weakness, breathing difficulty
Prevention: safe sex practices, needle safety, blood screening, antiretroviral prophylaxis (PrEP/PEP)
Contagious Period: lifelong (as long as virus present, reduced with treatment)
Special Notes: antiretroviral therapy (ART) allows long healthy life, early diagnosis is critical, undetectable viral load reduces transmission risk
===
Disease: Measles
Aliases: rubeola
Description: highly contagious viral infection characterized by fever, respiratory symptoms, and a widespread red rash
Cause: measles virus (a paramyxovirus)
Transmission: airborne droplets from coughing or sneezing, direct contact with infected secretions
Risk Groups: unvaccinated individuals, children, travelers to endemic areas, immunocompromised individuals
Incubation Period: 7-14 days
Symptoms:
- high fever
- cough
- runny nose
- conjunctivitis (red eyes)
- tiny white spots inside the mouth (Koplik spots)
- red blotchy rash starting on face and spreading downward
Progression:
1. exposure to virus via respiratory droplets
2. viral replication in respiratory tract
3. onset of fever and respiratory symptoms
4. development of rash spreading across body
Common Locations: respiratory tract, skin, eyes, mouth
Duration: 1-2 weeks
Severity: moderate_to_severe
Complications: pneumonia, encephalitis, ear infection, diarrhea, death in severe cases
Home Remedy: rest, hydration, fever management, supportive care
Avoid: contact with infected individuals, skipping vaccination, crowded environments during outbreaks
When to See a Doctor: high fever, rash, exposure history, breathing difficulty
Emergency Signs: difficulty breathing, seizures, confusion, severe dehydration
Prevention: MMR vaccination, isolation of infected individuals, hygiene measures
Contagious Period: about 4 days before rash to 4 days after rash onset
Special Notes: highly contagious with high transmission rate, vaccination is highly effective, no specific antiviral treatment available
===
Disease: Mumps
Aliases: epidemic parotitis
Description: contagious viral infection affecting the salivary glands, leading to swelling, fever, and systemic symptoms
Cause: mumps virus (a paramyxovirus)
Transmission: respiratory droplets, direct contact with saliva, sharing contaminated items
Risk Groups: unvaccinated individuals, children, students in close-contact settings, healthcare workers, travelers to endemic areas
Incubation Period: 12-25 days (commonly 16-18 days)
Symptoms:
- swollen salivary glands (puffy cheeks, tender jaw)
- fever
- headache
- muscle aches
- fatigue
- loss of appetite
Progression:
1. exposure to virus via respiratory droplets or saliva
2. viral replication in upper respiratory tract
3. spread to salivary glands causing swelling
4. recovery or rare complications affecting other organs
Common Locations: salivary glands (parotid), respiratory tract, testes, ovaries, pancreas, brain
Duration: 1-2 weeks
Severity: mild_to_moderate
Complications: meningitis, encephalitis, orchitis, oophoritis, pancreatitis, hearing loss, miscarriage
Home Remedy: rest, hydration, pain relief, warm or cold compress on glands, soft diet
Avoid: close contact with others, sharing utensils, public exposure during contagious period
When to See a Doctor: swelling of glands, high fever, severe headache, testicular pain, persistent symptoms
Emergency Signs: seizures, stiff neck, confusion, severe headache, drowsiness
Prevention: MMR vaccination, hygiene practices, isolation during infection
Contagious Period: a few days before swelling until about 5 days after onset
Special Notes: immunity usually lifelong after infection, vaccinated individuals may still get mild disease, no specific antiviral treatment
===
Disease: Rubella
Aliases: German measles
Description: contagious viral infection characterized by mild fever and rash, but can cause severe birth defects if contracted during pregnancy
Cause: rubella virus (a togavirus)
Transmission: airborne droplets from coughing or sneezing, direct contact, mother-to-fetus during pregnancy
Risk Groups: unvaccinated individuals, pregnant women, travelers to endemic regions, children
Incubation Period: 14-21 days
Symptoms:
- mild fever
- rash starting on face and spreading to body
- sore throat
- swollen lymph nodes
- joint pain (especially in adult women)
- fatigue
Progression:
1. exposure to virus via respiratory droplets
2. viral replication in respiratory tract
3. spread through bloodstream
4. development of rash and mild systemic symptoms
Common Locations: skin, lymph nodes, respiratory tract, fetus (in congenital infection)
Duration: 1-2 weeks
Severity: mild_to_moderate
Complications: arthritis, encephalitis (rare), congenital rubella syndrome, miscarriage, birth defects
Home Remedy: rest, hydration, fever management with acetaminophen
Avoid: contact with pregnant women, skipping vaccination, crowded environments during infection
When to See a Doctor: rash with fever, exposure history, pregnancy exposure
Emergency Signs: severe headache, confusion, neurological symptoms
Prevention: MMR vaccination, isolation during infection, hygiene practices
Contagious Period: about 1 week before rash to 7 days after rash onset
Special Notes: often mild in children but dangerous in pregnancy, lifelong immunity after infection or vaccination, no specific antiviral treatment
===
Disease: Chickenpox
Aliases: varicella, chicken pox
Description: highly contagious viral infection causing itchy fluid-filled blisters, primarily affecting children but possible at any age
Cause: varicella-zoster virus (VZV) infecting skin and nerve tissue
Transmission: airborne droplets, direct contact with blister fluid, contact with shingles lesions, highly infectious before rash appears
Risk Groups: unvaccinated individuals, children, pregnant women, newborns, adolescents, adults, immunocompromised individuals
Incubation Period: 10-21 days
Symptoms:
- fever, fatigue, headache, loss of appetite before rash
- itchy red rash progressing to fluid-filled blisters
- blisters that scab over within about a week
- rash starting on chest, back, face and spreading across body
- up to hundreds of lesions in severe cases
Progression:
1. initial mild fever and fatigue
2. appearance of red spots on skin
3. formation of fluid-filled blisters
4. blisters crust over and heal within 1-2 weeks
Common Locations: face, chest, back, scalp, arms, legs, mouth
Duration: 4-7 days (rash phase)
Severity: mild_to_moderate
Complications: bacterial skin infection, pneumonia, encephalitis, dehydration, fetal complications, severe disease in adults
Home Remedy: rest, hydration, soothing lotions, oatmeal baths, antihistamines for itching
Avoid: scratching, aspirin (risk of Reye syndrome), ibuprofen in some cases, close contact with others
When to See a Doctor: high fever, severe rash, breathing difficulty, immunocompromised status, pregnancy exposure
Emergency Signs: difficulty breathing, severe dehydration, neurological symptoms, persistent vomiting, altered consciousness
Prevention: varicella vaccination (2 doses), avoiding contact with infected individuals, hygiene practices
Contagious Period: 1-2 days before rash until all lesions crust over
Special Notes: virus remains dormant and can reactivate as shingles later in life, vaccination greatly reduces severity and complications
===
Disease: Rabies
Aliases: hydrophobia (advanced stage)
Description: fatal viral infection affecting the central nervous system leading to brain inflammation and death if untreated before symptom onset
Cause: rabies virus (a lyssavirus) transmitted through infected animal saliva
Transmission: bites or scratches from infected animals, saliva contact with open wounds or mucous membranes
Risk Groups: people exposed to wild animals, travelers to endemic regions, veterinarians, animal handlers, outdoor workers
Incubation Period: weeks to months (variable depending on exposure site)
Symptoms:
- fever, headache, weakness (early stage)
- tingling or itching at bite site
- anxiety, confusion, agitation
- hallucinations
- excessive salivation
- fear of water (hydrophobia)
Progression:
1. virus enters body through bite or scratch
2. travels along nerves to central nervous system
3. causes brain inflammation and neurological symptoms
4. progression to coma and death if untreated
Common Locations: brain, central nervous system, salivary glands
Duration: days to weeks after symptom onset (rapid progression)
Severity: severe
Complications: encephalitis, paralysis, respiratory failure, death
Home Remedy: none
Avoid: contact with wild or unvaccinated animals, delayed post-exposure treatment
When to See a Doctor: any animal bite or suspected exposure, especially from wild animals
Emergency Signs: confusion, hallucinations, difficulty swallowing, paralysis, seizures
Prevention: rabies vaccination (pre- and post-exposure), wound cleaning, avoiding animal contact, vaccinating pets
Contagious Period: present in saliva of infected animals, human-to-human transmission extremely rare
Special Notes: nearly 100% fatal after symptom onset, post-exposure prophylaxis (PEP) is highly effective if given early, bats are a major source in some regions
===
Disease: Leprosy
Aliases: Hansen disease
Description: chronic bacterial infection affecting skin, peripheral nerves, and mucous membranes leading to progressive damage and disability if untreated
Cause: Mycobacterium leprae bacteria infecting skin and nerves
Transmission: prolonged close contact via respiratory droplets or nasal secretions from untreated individuals
Risk Groups: people in endemic regions, close contacts of infected individuals, children, immunocompromised individuals
Incubation Period: months to years (typically 3-5 years)
Symptoms:
- light-colored or reddish skin lesions with reduced sensation
- numbness in hands, arms, feet, or legs
- muscle weakness
- thickened nerves
- non-healing skin sores
Progression:
1. infection via respiratory exposure
2. slow bacterial growth affecting skin and nerves
3. development of sensory loss and skin lesions
4. nerve damage and disability if untreated
Common Locations: skin, peripheral nerves, face, hands, feet
Duration: chronic, may last years without treatment
Severity: mild_to_severe
Complications: nerve damage, loss of sensation, muscle weakness, deformities, disability
Home Remedy: none
Avoid: delayed treatment, close contact without precautions, neglect of symptoms
When to See a Doctor: persistent skin lesions, numbness, weakness, contact with infected person
Emergency Signs: severe nerve damage, loss of function, secondary infections
Prevention: early diagnosis and treatment, avoiding prolonged close contact with untreated cases
Contagious Period: low, mainly in untreated individuals (reduced after starting treatment)
Special Notes: treatable with multi-drug therapy, early treatment prevents disability, stigma associated with disease despite low infectivity
===
Disease: Whooping Cough
Aliases: pertussis
Description: highly contagious bacterial respiratory infection causing severe coughing fits and airway inflammation
Cause: Bordetella pertussis bacteria damaging respiratory cilia and releasing toxins
Transmission: airborne droplets from coughing or sneezing, close contact in shared airspace
Risk Groups: infants under 1 year, unvaccinated individuals, elderly, people with asthma or weakened immune systems
Incubation Period: 5-10 days (up to 21 days)
Symptoms:
- mild cough and runny nose (early stage)
- severe coughing fits (paroxysmal cough)
- whooping sound during inhalation after coughing
- vomiting after coughing
- exhaustion after coughing episodes
Progression:
1. initial mild cold-like symptoms (catarrhal stage)
2. progression to severe coughing fits (paroxysmal stage)
3. gradual recovery with reduced coughing (convalescent stage)
4. prolonged cough may persist for weeks
Common Locations: upper respiratory tract, lungs
Duration: weeks to months (often called 100-day cough)
Severity: moderate_to_severe
Complications: pneumonia, apnea (in infants), seizures, brain damage, death (especially in infants)
Home Remedy: rest, hydration, humidified air, supportive care
Avoid: close contact with others, exposure of infants, delayed treatment
When to See a Doctor: persistent severe cough, coughing fits, vomiting after coughing, infant symptoms
Emergency Signs: difficulty breathing, apnea, bluish skin, seizures, severe exhaustion
Prevention: vaccination (DTaP/Tdap), antibiotics after exposure, hygiene practices
Contagious Period: from early symptoms up to 2 weeks after coughing begins (shorter with antibiotics)
Special Notes: early antibiotic treatment reduces severity and spread, immunity not lifelong, infants are at highest risk
===
Disease: Tetanus
Aliases: lockjaw
Description: serious bacterial infection affecting the nervous system causing muscle stiffness and painful spasms
Cause: Clostridium tetani bacteria producing a neurotoxin that affects motor nerves
Transmission: contamination of wounds with bacterial spores from soil, dust, or animal waste (not person-to-person)
Risk Groups: unvaccinated individuals, people with deep or contaminated wounds, intravenous drug users, newborns in unsanitary conditions
Incubation Period: 3-21 days (can range from days to months)
Symptoms:
- muscle stiffness and spasms (especially jaw - lockjaw)
- difficulty swallowing
- neck and back stiffness
- abdominal muscle rigidity
- painful muscle contractions
Progression:
1. spores enter body through wound
2. bacteria produce toxin in anaerobic conditions
3. toxin spreads via nerves to central nervous system
4. muscle rigidity and severe spasms develop
Common Locations: nervous system, muscles, jaw, neck, back
Duration: weeks to months (recovery may be prolonged)
Severity: severe
Complications: respiratory failure, fractures from spasms, pneumonia, autonomic instability, death
Home Remedy: none
Avoid: untreated wounds, poor hygiene, missing vaccinations
When to See a Doctor: deep or contaminated wounds, unknown vaccination status, muscle stiffness or spasms
Emergency Signs: difficulty breathing, severe spasms, inability to swallow, generalized rigidity
Prevention: vaccination (DTaP/Tdap), booster doses every 10 years, proper wound care, sterile practices
Contagious Period: none
Special Notes: not spread person-to-person, neonatal tetanus occurs in unsanitary conditions, early vaccination is highly effective
===
Disease: Diphtheria
Aliases: none
Description: serious bacterial infection caused by toxin-producing bacteria leading to tissue damage, especially in the respiratory tract
Cause: Corynebacterium diphtheriae producing diphtheria toxin
Transmission: airborne droplets from coughing or sneezing, contact with infected wounds or contaminated objects
Risk Groups: unvaccinated individuals, close contacts of infected persons, travelers to endemic regions, people with poor immunity
Incubation Period: 2-5 days
Symptoms:
- sore throat and difficulty swallowing
- fever and weakness
- thick gray membrane in throat or nose
- swollen neck (bull neck appearance)
- breathing difficulty
Progression:
1. exposure to bacteria via droplets or contact
2. bacterial colonization in respiratory tract or skin
3. toxin production causing tissue damage
4. systemic complications if untreated
Common Locations: throat, nose, skin
Duration: weeks (with treatment recovery improves)
Severity: severe
Complications: airway obstruction, myocarditis, nerve damage, respiratory failure, death
Home Remedy: none
Avoid: close contact with infected individuals, skipping vaccination, delayed treatment
When to See a Doctor: sore throat with membrane, breathing difficulty, exposure history
Emergency Signs: severe breathing difficulty, airway blockage, heart problems, neurological symptoms
Prevention: vaccination (DTaP/Tdap), antibiotics for close contacts, hygiene practices
Contagious Period: until about 48 hours after starting antibiotics
Special Notes: toxin causes most damage rather than bacteria itself, early antitoxin treatment is critical, vaccination highly effective
===
Disease: Japanese Encephalitis
Aliases: JE
Description: mosquito-borne viral infection causing inflammation of the brain, potentially leading to severe neurological damage or death
Cause: Japanese encephalitis virus (a flavivirus)
Transmission: bite of infected mosquitoes (mainly Culex species), not spread person-to-person
Risk Groups: people living in rural Asia, long-term travelers to endemic regions, outdoor workers, laboratory personnel
Incubation Period: 5-15 days
Symptoms:
- fever and headache (mild cases)
- neck stiffness
- seizures
- confusion or altered consciousness
- coma in severe cases
Progression:
1. infection via mosquito bite
2. viral replication in bloodstream
3. invasion of central nervous system
4. encephalitis with neurological symptoms
Common Locations: brain, central nervous system
Duration: days to weeks (neurological recovery may take longer)
Severity: severe
Complications: brain damage, seizures, paralysis, coma, permanent disability, death
Home Remedy: none
Avoid: mosquito bites, travel without protection, staying in high-risk rural areas without precautions
When to See a Doctor: fever with neurological symptoms, travel history to endemic areas
Emergency Signs: seizures, coma, severe confusion, inability to wake
Prevention: JE vaccination, mosquito control, insect repellents, protective clothing
Contagious Period: none (not transmitted person-to-person)
Special Notes: most infections are asymptomatic, severe cases have high mortality, survivors may have long-term neurological deficits
===
Disease: Leishmaniasis
Aliases: kala-azar, cutaneous leishmaniasis, visceral leishmaniasis
Description: parasitic disease transmitted by sandfly bites affecting skin, mucous membranes, or internal organs depending on type
Cause: Leishmania protozoa parasites infecting immune cells
Transmission: bite of infected female sandflies
Risk Groups: people in tropical and subtropical regions, travelers to endemic areas, immunocompromised individuals, military personnel
Incubation Period: weeks to months (commonly 2-8 months for visceral form)
Symptoms:
- skin sores or ulcers (cutaneous form)
- nasal congestion, nosebleeds, or mucosal ulcers
- fever and weight loss (visceral form)
- fatigue and weakness
- enlarged spleen and liver
- night sweats
Progression:
1. parasite transmitted via sandfly bite
2. infection of macrophages in skin or blood
3. localized or systemic spread depending on type
4. immune system damage and organ involvement (visceral form)
Common Locations: skin, mucous membranes, liver, spleen, bone marrow
Duration: weeks to years depending on form and treatment
Severity: mild_to_severe
Complications: disfigurement, secondary infections, organ failure, bleeding, death (especially visceral type)
Home Remedy: none
Avoid: sandfly exposure, travel without protection, delayed treatment
When to See a Doctor: persistent skin sores, fever, weight loss, travel to endemic areas
Emergency Signs: severe weakness, bleeding, high fever, organ enlargement, signs of systemic infection
Prevention: insect repellents, protective clothing, bed nets, environmental control of sandflies
Contagious Period: none (not spread person-to-person)
Special Notes: visceral form (kala-azar) is most severe, early treatment improves outcomes, no widely available vaccine
===
Disease: Filariasis
Aliases: filaria, lymphatic filariasis, elephantiasis
Description: parasitic infection caused by filarial worms transmitted by mosquitoes, leading to lymphatic system damage and swelling
Cause: filarial nematodes (Wuchereria bancrofti, Brugia malayi, Brugia timori)
Transmission: bite of infected mosquitoes transferring larvae into bloodstream
Risk Groups: people in tropical and subtropical regions, individuals in areas with poor sanitation, long-term residents of endemic regions
Incubation Period: months to years
Symptoms:
- fever and lymph node inflammation (early stage)
- swelling of limbs (lymphedema)
- thickened skin (elephantiasis)
- pain in affected areas
- hydrocele (swelling of scrotum in males)
Progression:
1. mosquito transmits larvae into bloodstream
2. larvae develop into adult worms in lymphatic vessels
3. blockage and inflammation of lymphatic system
4. chronic swelling and tissue changes
Common Locations: lymphatic system, limbs, genitals
Duration: chronic, may last years or lifelong
Severity: moderate_to_severe
Complications: elephantiasis, disability, secondary infections, social stigma
Home Remedy: hygiene care, limb elevation, skin care to prevent infection
Avoid: mosquito exposure, poor hygiene, untreated infection
When to See a Doctor: persistent swelling, lymph node pain, fever in endemic areas
Emergency Signs: severe swelling with infection, high fever, rapid worsening symptoms
Prevention: mosquito control, insect repellents, protective clothing, mass drug administration programs
Contagious Period: none (not directly person-to-person)
Special Notes: long-term infection leads to irreversible changes, early treatment prevents progression, common in endemic tropical regions
===
Disease: Plague
Aliases: bubonic plague, septicemic plague, pneumonic plague
Description: severe bacterial infection transmitted from animals to humans, historically associated with epidemics and capable of causing rapid systemic illness
Cause: Yersinia pestis bacteria infecting blood and lymphatic system
Transmission: flea bites from infected rodents, contact with infected animal tissues or fluids, inhalation of infectious droplets (pneumonic form)
Risk Groups: people in rural or semi-rural areas, hunters, veterinarians, individuals exposed to rodents or fleas, people in outbreak regions
Incubation Period: 1-7 days
Symptoms:
- fever and chills
- swollen and painful lymph nodes (buboes)
- weakness and fatigue
- abdominal pain
- cough and bloody sputum (pneumonic form)
- skin turning black due to tissue death (severe cases)
Progression:
1. transmission via flea bite, contact, or inhalation
2. bacterial spread to lymph nodes or bloodstream
3. development of bubonic, septicemic, or pneumonic forms
4. rapid systemic infection and potential organ failure
Common Locations: lymph nodes, blood, lungs
Duration: days to weeks (rapid progression if untreated)
Severity: severe
Complications: sepsis, respiratory failure, organ failure, shock, death
Home Remedy: none
Avoid: flea exposure, contact with infected animals, handling animals without protection
When to See a Doctor: fever with swollen lymph nodes, recent exposure to rodents or fleas, respiratory symptoms after exposure
Emergency Signs: severe breathing difficulty, shock, confusion, bleeding, rapid deterioration
Prevention: flea control, avoiding rodent exposure, protective equipment when handling animals, early antibiotic treatment
Contagious Period: pneumonic plague can spread person-to-person via droplets
Special Notes: requires urgent antibiotic treatment, pneumonic form is highly contagious, outbreaks linked to rodent populations and fleas
===
Disease: Zika Virus Infection
Aliases: Zika, Zika fever
Description: mosquito-borne viral infection usually causing mild illness but associated with serious complications during pregnancy and rare neurological disorders
Cause: Zika virus (a flavivirus)
Transmission: bite of infected Aedes mosquitoes, sexual transmission, mother-to-fetus during pregnancy, rarely via blood transfusion
Risk Groups: people in tropical and subtropical regions, travelers to endemic areas, pregnant women, sexual partners of infected individuals
Incubation Period: 3-14 days
Symptoms:
- fever
- rash
- headache
- joint and muscle pain
- red eyes (conjunctivitis)
- fatigue
Progression:
1. infection via mosquito bite or other transmission
2. viral replication in bloodstream
3. mild symptomatic phase or asymptomatic infection
4. recovery or rare complications
Common Locations: blood, nervous system, fetus (in pregnancy)
Duration: several days to 1 week
Severity: mild
Complications: congenital birth defects (microcephaly), Guillain-Barre syndrome, neurological complications, pregnancy loss
Home Remedy: rest, hydration, supportive care
Avoid: mosquito exposure, unprotected sex, travel to endemic areas during pregnancy
When to See a Doctor: pregnancy with exposure, persistent symptoms, neurological signs
Emergency Signs: paralysis, severe neurological symptoms, confusion, difficulty breathing
Prevention: mosquito control, insect repellents, protective clothing, safe sex practices
Contagious Period: present in blood and bodily fluids for days to weeks (sexual transmission possible longer)
Special Notes: often asymptomatic, major concern is during pregnancy, no vaccine or specific antiviral treatment available
===
Disease: H1N1 Swine Flu
Aliases: swine flu, H1N1 influenza
Description: viral respiratory infection caused by a subtype of influenza A virus leading to flu-like symptoms and occasional severe illness
Cause: influenza A (H1N1) virus infecting the respiratory tract
Transmission: airborne droplets from coughing or sneezing, contact with contaminated surfaces, close contact with infected individuals
Risk Groups: children, elderly, pregnant women, individuals with chronic diseases, immunocompromised individuals
Incubation Period: 1-4 days
Symptoms:
- fever and chills
- cough and sore throat
- runny or stuffy nose
- body aches and fatigue
- headache
- nausea, vomiting, or diarrhea (sometimes)
Progression:
1. exposure to virus via droplets or contact
2. viral replication in respiratory tract
3. onset of flu-like symptoms
4. recovery or progression to complications in high-risk individuals
Common Locations: nose, throat, lungs
Duration: 1-2 weeks
Severity: mild_to_severe
Complications: pneumonia, respiratory failure, worsening of chronic conditions, death in severe cases
Home Remedy: rest, hydration, over-the-counter medications for fever and pain relief
Avoid: close contact with infected individuals, crowded places, smoking, ignoring symptoms
When to See a Doctor: high fever, breathing difficulty, worsening symptoms, high-risk individuals
Emergency Signs: difficulty breathing, chest pain, confusion, bluish skin, persistent vomiting
Prevention: vaccination, hand hygiene, mask use, avoiding close contact with infected individuals
Contagious Period: about 1 day before symptoms to 5-7 days after onset
Special Notes: similar to seasonal flu but may affect younger populations more, antiviral drugs may reduce severity if given early
===
Disease: Rotavirus Infection
Aliases: rotavirus gastroenteritis
Description: highly contagious viral infection causing severe diarrhea and vomiting, primarily affecting infants and young children
Cause: rotavirus infecting the intestinal lining
Transmission: fecal-oral route via contaminated hands, surfaces, food, or water
Risk Groups: infants and young children, unvaccinated children, daycare attendees, immunocompromised individuals
Incubation Period: 1-3 days
Symptoms:
- severe watery diarrhea
- vomiting
- fever
- abdominal pain
- dehydration
Progression:
1. ingestion of virus through contaminated sources
2. infection of intestinal cells
3. onset of vomiting and diarrhea
4. recovery or dehydration complications if untreated
Common Locations: small intestine
Duration: 3-8 days
Severity: mild_to_severe
Complications: severe dehydration, electrolyte imbalance, hospitalization, death (rare with treatment)
Home Remedy: oral rehydration, fluids, rest, zinc supplementation in children
Avoid: poor hygiene, contaminated food or water, delayed rehydration
When to See a Doctor: persistent vomiting, signs of dehydration, inability to drink fluids
Emergency Signs: severe dehydration, lethargy, sunken eyes, minimal urine output, rapid breathing
Prevention: rotavirus vaccination, hand hygiene, sanitation, safe food practices
Contagious Period: during illness and for several days after recovery
Special Notes: leading cause of severe diarrhea in children globally, vaccination significantly reduces severe cases
===
Disease: Norovirus Infection
Aliases: norovirus gastroenteritis, stomach bug
Description: highly contagious viral infection causing acute gastroenteritis with vomiting and diarrhea
Cause: norovirus infecting the stomach and intestines
Transmission: fecal-oral route, contaminated food or water, direct contact with infected person, contact with contaminated surfaces
Risk Groups: children under 5, elderly, immunocompromised individuals, people in crowded settings (schools, cruise ships, hospitals)
Incubation Period: 12-48 hours
Symptoms:
- vomiting
- diarrhea
- nausea
- stomach pain
- fever
- headache and body aches
- dehydration signs (dry mouth, dizziness)
Progression:
1. ingestion of virus via contaminated food, water, or contact
2. viral replication in gastrointestinal tract
3. sudden onset of vomiting and diarrhea
4. recovery within a few days or complications due to dehydration
Common Locations: stomach, intestines
Duration: 1-3 days
Severity: mild_to_moderate
Complications: dehydration, electrolyte imbalance, hospitalization (especially in high-risk groups)
Home Remedy: oral rehydration, fluids, rest, electrolyte solutions
Avoid: contaminated food, raw shellfish, poor hygiene, close contact during illness
When to See a Doctor: persistent vomiting, dehydration signs, inability to retain fluids
Emergency Signs: severe dehydration, confusion, minimal urination, extreme weakness
Prevention: hand hygiene, proper food handling, surface disinfection, avoiding contact when infected
Contagious Period: during illness and up to 2 weeks after recovery
Special Notes: multiple strains allow repeated infections, antibiotics ineffective, highly outbreak-prone in closed environments
===
Disease: Amoebiasis
Aliases: amebiasis, amoebic dysentery
Description: parasitic intestinal infection causing diarrhea and colitis, sometimes spreading to other organs like the liver
Cause: Entamoeba histolytica protozoan parasite
Transmission: fecal-oral route via contaminated food or water, poor sanitation, person-to-person contact
Risk Groups: people in areas with poor sanitation, travelers to endemic regions, individuals with unsafe food or water exposure, immunocompromised individuals
Incubation Period: 1-4 weeks
Symptoms:
- diarrhea (may be bloody)
- abdominal pain and cramping
- fever
- nausea
- weight loss
- fatigue
Progression:
1. ingestion of parasite cysts via contaminated sources
2. cysts release active trophozoites in intestine
3. invasion of intestinal lining causing ulcers
4. possible spread to liver or other organs
Common Locations: intestines (colon), liver
Duration: weeks to months if untreated
Severity: mild_to_severe
Complications: liver abscess, intestinal perforation, severe colitis, spread to lungs or brain (rare)
Home Remedy: hydration, proper nutrition, hygiene practices
Avoid: contaminated food or water, poor sanitation, delayed treatment
When to See a Doctor: persistent diarrhea, blood in stool, abdominal pain, travel history
Emergency Signs: severe abdominal pain, high fever, signs of liver involvement, dehydration
Prevention: safe drinking water, proper sanitation, hand hygiene, food safety
Contagious Period: while parasite cysts are shed in stool
Special Notes: can be asymptomatic in some individuals, requires antiparasitic treatment, common in developing regions
===
Disease: Giardiasis
Aliases: giardia infection, beaver fever
Description: intestinal parasitic infection causing diarrhea and digestive disturbances due to protozoan infection
Cause: Giardia lamblia (Giardia intestinalis) protozoan parasite infecting the small intestine
Transmission: fecal-oral route via contaminated water, food, surfaces, or person-to-person contact
Risk Groups: travelers, campers, children in daycare, people drinking untreated water, individuals with poor sanitation
Incubation Period: 1-3 weeks
Symptoms:
- diarrhea (often greasy or foul-smelling)
- abdominal cramps and bloating
- nausea
- fatigue
- weight loss
- gas and indigestion
Progression:
1. ingestion of cysts from contaminated sources
2. parasites multiply in small intestine
3. interference with nutrient absorption
4. symptoms develop and may become chronic if untreated
Common Locations: small intestine
Duration: 2-6 weeks (may become chronic)
Severity: mild_to_moderate
Complications: dehydration, malnutrition, lactose intolerance, chronic gastrointestinal issues
Home Remedy: hydration, proper nutrition, hygiene practices
Avoid: untreated water, poor hygiene, contaminated food, close contact during infection
When to See a Doctor: persistent diarrhea, weight loss, abdominal discomfort, travel history
Emergency Signs: severe dehydration, inability to retain fluids, extreme weakness
Prevention: safe drinking water, hand hygiene, proper sanitation, avoiding contaminated sources
Contagious Period: while cysts are present in stool
Special Notes: common in hikers and campers, can cause prolonged digestive issues, treatable with antiparasitic medication
===
Disease: Cryptosporidiosis
Aliases: crypto infection
Description: parasitic diarrheal disease caused by Cryptosporidium affecting the intestines and leading to watery diarrhea
Cause: Cryptosporidium protozoa infecting intestinal epithelial cells
Transmission: fecal-oral route via contaminated water, food, surfaces, or person-to-person contact
Risk Groups: children, immunocompromised individuals, swimmers in contaminated water, people in areas with poor sanitation, animal handlers
Incubation Period: 2-10 days (commonly around 7 days)
Symptoms:
- watery diarrhea
- abdominal cramps
- nausea and vomiting
- fever
- dehydration
- weight loss
Progression:
1. ingestion of oocysts from contaminated sources
2. parasites infect intestinal lining
3. inflammation and fluid secretion causing diarrhea
4. recovery or prolonged illness in immunocompromised individuals
Common Locations: small intestine
Duration: 1-2 weeks (longer in immunocompromised individuals)
Severity: mild_to_severe
Complications: severe dehydration, malnutrition, chronic diarrhea, life-threatening illness in immunocompromised patients
Home Remedy: oral rehydration, fluids, rest, proper nutrition
Avoid: contaminated water (including pools), poor hygiene, untreated water consumption
When to See a Doctor: persistent diarrhea, dehydration, symptoms in immunocompromised individuals
Emergency Signs: severe dehydration, inability to drink fluids, extreme weakness, prolonged illness
Prevention: safe drinking water, hand hygiene, proper sanitation, avoiding swallowing pool or recreational water
Contagious Period: while oocysts are shed in stool
Special Notes: resistant to chlorine in pools, can cause severe disease in HIV/AIDS patients, no fully effective treatment in severe cases
===
Disease: Salmonella Infection
Aliases: salmonellosis
Description: bacterial gastrointestinal infection causing diarrhea, fever, and abdominal cramps due to ingestion of contaminated food or water
Cause: Salmonella bacteria (non-typhoidal species) infecting the intestines
Transmission: ingestion of contaminated food (e.g., undercooked poultry, eggs), water, or contact with infected animals or surfaces
Risk Groups: children, elderly, immunocompromised individuals, people consuming undercooked food, travelers
Incubation Period: 6-48 hours
Symptoms:
- diarrhea (sometimes bloody)
- fever
- abdominal cramps
- nausea and vomiting
- headache
- dehydration
Progression:
1. ingestion of contaminated food or water
2. bacteria invade intestinal lining
3. onset of gastrointestinal symptoms
4. recovery or spread to bloodstream in severe cases
Common Locations: intestines, bloodstream (in severe cases)
Duration: 4-7 days
Severity: mild_to_moderate
Complications: dehydration, bacteremia, reactive arthritis, severe infection in high-risk groups
Home Remedy: hydration, oral rehydration solutions, rest
Avoid: undercooked food, poor hygiene, cross-contamination, unnecessary antibiotics
When to See a Doctor: severe diarrhea, high fever, blood in stool, dehydration signs
Emergency Signs: severe dehydration, persistent vomiting, confusion, signs of bloodstream infection
Prevention: proper food handling, thorough cooking, hand hygiene, avoiding contaminated food or water
Contagious Period: while bacteria are present in stool
Special Notes: usually self-limiting, antibiotics reserved for severe cases, proper food safety reduces risk
===
Disease: Shigellosis
Aliases: bacillary dysentery, shigella infection
Description: bacterial intestinal infection causing diarrhea, often with blood and mucus, due to inflammation of the colon
Cause: Shigella bacteria (e.g., Shigella dysenteriae, Shigella flexneri)
Transmission: fecal-oral route via contaminated food, water, surfaces, or person-to-person contact
Risk Groups: children, daycare attendees, travelers to endemic regions, people with poor sanitation, immunocompromised individuals
Incubation Period: 1-3 days
Symptoms:
- diarrhea (often bloody or with mucus)
- abdominal cramps
- fever
- nausea and vomiting
- tenesmus (feeling of incomplete bowel evacuation)
Progression:
1. ingestion of bacteria through contaminated sources
2. invasion of intestinal lining (colon)
3. inflammation and ulceration of mucosa
4. onset of dysentery symptoms
Common Locations: colon (large intestine)
Duration: 5-7 days (may last longer)
Severity: mild_to_severe
Complications: dehydration, seizures (especially in children), hemolytic uremic syndrome, toxic megacolon
Home Remedy: hydration, oral rehydration, rest
Avoid: poor hygiene, contaminated food or water, close contact during illness
When to See a Doctor: bloody diarrhea, high fever, dehydration, persistent symptoms
Emergency Signs: severe dehydration, seizures, confusion, inability to drink fluids
Prevention: hand hygiene, safe food and water practices, sanitation, avoiding contaminated sources
Contagious Period: while bacteria are present in stool (highly contagious)
Special Notes: very low infectious dose, spreads easily in crowded settings, antibiotics used in severe cases
===
Disease: Hand-Foot-Mouth Disease
Aliases: HFMD, coxsackievirus infection
Description: common contagious viral infection primarily affecting children, characterized by sores in the mouth and rash on hands and feet
Cause: coxsackievirus A16 (most common), other enteroviruses
Transmission: respiratory droplets, contact with contaminated surfaces, contact with stool or blister fluid of infected person
Risk Groups: children under 10 years, daycare attendees, close contacts, occasionally adults
Incubation Period: 3-7 days
Symptoms:
- fever
- sore throat
- loss of appetite
- painful mouth ulcers
- rash with small blisters on hands, feet, and diaper area
- headache
Progression:
1. exposure to virus via droplets or contact
2. viral replication in throat and intestines
3. onset of fever and sore throat
4. development of mouth sores and skin rash
Common Locations: mouth, hands, feet, throat, sometimes buttocks
Duration: 5-7 days
Severity: mild
Complications: dehydration, febrile seizures, rare neurological complications
Home Remedy: hydration, rest, pain relief (acetaminophen or ibuprofen), saltwater rinses
Avoid: close contact with infected individuals, sharing items, poor hygiene, aspirin in children
When to See a Doctor: persistent fever, difficulty swallowing, dehydration signs, worsening symptoms
Emergency Signs: seizures, severe dehydration, lethargy, inability to drink fluids
Prevention: hand hygiene, disinfecting surfaces, avoiding contact during infection
Contagious Period: most contagious during first week, may spread for weeks via stool
Special Notes: antibiotics ineffective, usually self-limiting, common in summer and early fall
===
Disease: Severe Acute Respiratory Syndrome
Aliases: SARS, SARS-CoV infection
Description: serious viral respiratory illness causing pneumonia and severe breathing problems with potential for outbreaks
Cause: SARS coronavirus (SARS-CoV)
Transmission: respiratory droplets, close person-to-person contact, contaminated surfaces, possible airborne spread in certain conditions
Risk Groups: healthcare workers, close contacts of infected individuals, travelers to outbreak regions, elderly
Incubation Period: 2-10 days
Symptoms:
- high fever
- dry cough
- shortness of breath
- headache
- muscle aches
- chills
Progression:
1. exposure to virus via respiratory droplets
2. viral replication in respiratory tract
3. onset of fever and flu-like symptoms
4. progression to pneumonia and respiratory distress in severe cases
Common Locations: lungs, respiratory tract
Duration: weeks (variable depending on severity)
Severity: severe
Complications: pneumonia, acute respiratory distress syndrome, respiratory failure, death
Home Remedy: rest, hydration, supportive care
Avoid: close contact with infected individuals, crowded environments, poor hygiene
When to See a Doctor: fever with respiratory symptoms, exposure history, worsening breathing
Emergency Signs: severe breathlessness, chest pain, confusion, low oxygen levels
Prevention: infection control measures, mask use, hygiene, isolation during outbreaks
Contagious Period: during symptomatic phase (especially after symptom onset)
Special Notes: outbreak occurred in early 2000s, controlled through public health measures, related to other coronaviruses like COVID-19
===
Disease: Middle East Respiratory Syndrome
Aliases: MERS, MERS-CoV infection
Description: severe viral respiratory illness caused by a coronavirus leading to pneumonia and high mortality, primarily reported in the Middle East
Cause: Middle East respiratory syndrome coronavirus (MERS-CoV)
Transmission: close contact with infected individuals, respiratory droplets, contact with infected camels or animal products
Risk Groups: people in or traveling to Middle Eastern regions, healthcare workers, elderly, immunocompromised individuals, those with chronic diseases
Incubation Period: 2-14 days
Symptoms:
- fever
- cough
- shortness of breath
- chest pain
- fatigue
- diarrhea (in some cases)
Progression:
1. exposure to virus via respiratory droplets or animal contact
2. viral replication in respiratory tract
3. onset of respiratory symptoms
4. progression to severe pneumonia and organ involvement in some cases
Common Locations: lungs, respiratory tract
Duration: weeks (depending on severity)
Severity: severe
Complications: pneumonia, acute respiratory distress syndrome, kidney failure, death
Home Remedy: rest, hydration, supportive care
Avoid: contact with infected individuals, raw camel products, poor hygiene, crowded healthcare settings during outbreaks
When to See a Doctor: fever with respiratory symptoms, recent travel to affected regions, contact with camels
Emergency Signs: severe breathing difficulty, organ failure signs, confusion, low oxygen levels
Prevention: infection control measures, hand hygiene, avoiding contact with camels, protective equipment in healthcare settings
Contagious Period: during symptomatic phase (especially close contact situations)
Special Notes: high fatality rate, limited human-to-human spread outside healthcare settings, no widely available vaccine
===
Disease: Hantavirus Infection
Aliases: hantavirus pulmonary syndrome, HPS
Description: rare but severe viral disease causing respiratory failure and systemic illness due to infection from rodent-borne viruses
Cause: hantavirus (family Hantaviridae) transmitted from infected rodents
Transmission: inhalation of aerosolized particles from rodent urine, droppings, or saliva, rarely through bites
Risk Groups: people exposed to rodents, rural residents, farmers, campers, construction workers, individuals cleaning rodent-infested areas
Incubation Period: 1-8 weeks
Symptoms:
- fever and chills
- muscle aches
- fatigue
- headache
- nausea and vomiting
- shortness of breath (later stage)
Progression:
1. exposure to contaminated rodent excreta
2. viral entry via respiratory tract
3. early flu-like symptoms
4. progression to severe respiratory distress and lung involvement
Common Locations: lungs, blood vessels
Duration: days to weeks (rapid progression in severe cases)
Severity: severe
Complications: respiratory failure, shock, organ failure, death
Home Remedy: none
Avoid: exposure to rodent-infested environments, inhaling dust in contaminated areas, poor hygiene
When to See a Doctor: flu-like symptoms after rodent exposure, breathing difficulty
Emergency Signs: severe breathlessness, low blood pressure, confusion, rapid deterioration
Prevention: rodent control, proper sanitation, protective measures when cleaning contaminated areas
Contagious Period: none (not typically spread person-to-person)
Special Notes: high fatality rate in severe cases, early recognition critical, more common in rural settings
===
Disease: Scrub Typhus
Aliases: bush typhus
Description: bacterial infection transmitted by mite larvae causing fever, rash, and systemic illness common in rural tropical regions
Cause: Orientia tsutsugamushi bacteria infecting endothelial cells
Transmission: bite of infected chiggers (larval mites), not spread person-to-person
Risk Groups: people in rural or forested areas, farmers, outdoor workers, travelers to endemic regions (Asia-Pacific)
Incubation Period: 6-21 days
Symptoms:
- fever and chills
- headache
- muscle aches
- rash
- eschar (black scab at bite site)
- swollen lymph nodes
Progression:
1. bite from infected chigger
2. bacterial spread through bloodstream
3. onset of fever and systemic symptoms
4. possible organ involvement if untreated
Common Locations: skin (bite site), blood vessels, lungs, liver
Duration: 1-3 weeks (longer if untreated)
Severity: moderate_to_severe
Complications: pneumonia, encephalitis, organ failure, bleeding disorders, death
Home Remedy: none
Avoid: exposure to mite-infested areas, sitting on grass without protection, poor hygiene
When to See a Doctor: fever with rash or eschar, travel or exposure history, persistent symptoms
Emergency Signs: severe breathing difficulty, confusion, organ failure signs, seizures
Prevention: protective clothing, insect repellents, avoiding dense vegetation, environmental control
Contagious Period: none (not transmitted person-to-person)
Special Notes: early antibiotic treatment is highly effective, eschar is a key diagnostic sign, common in the tsutsugamushi triangle
===
Disease: Hypertension
Aliases: high blood pressure
Description: chronic medical condition characterized by persistently elevated arterial blood pressure increasing risk of cardiovascular diseases
Cause: multifactorial, including genetic factors, high salt intake, obesity, stress, and underlying conditions
Transmission: none
Risk Groups: older adults, obese individuals, smokers, people with family history, sedentary lifestyle, high-salt diet
Incubation Period: none
Symptoms:
- often asymptomatic (silent condition)
- headaches (in severe cases)
- dizziness
- blurred vision
- chest pain (in advanced cases)
Progression:
1. gradual increase in blood pressure
2. sustained elevation damages blood vessels
3. strain on heart and organs
4. development of complications over time
Common Locations: blood vessels, heart, kidneys, brain
Duration: chronic (lifelong if unmanaged)
Severity: mild_to_severe
Complications: heart attack, stroke, kidney failure, vision loss, heart failure
Home Remedy: lifestyle changes (low-salt diet, exercise, weight control, stress reduction)
Avoid: high salt intake, smoking, excessive alcohol, sedentary lifestyle
When to See a Doctor: consistently high blood pressure readings, headaches, dizziness, routine screening
Emergency Signs: hypertensive crisis (very high BP), chest pain, severe headache, confusion, vision changes
Prevention: healthy diet, regular exercise, maintaining healthy weight, limiting alcohol, regular monitoring
Contagious Period: none
Special Notes: often called silent killer, requires regular monitoring, managed with lifestyle changes and medications
===
Disease: Coronary Artery Disease
Aliases: CAD, ischemic heart disease
Description: chronic condition caused by narrowing or blockage of coronary arteries leading to reduced blood flow to the heart muscle
Cause: atherosclerosis (plaque buildup of cholesterol, fat, and other substances in artery walls)
Transmission: none
Risk Groups: older adults, smokers, diabetics, obese individuals, people with high cholesterol or hypertension, sedentary lifestyle
Incubation Period: none
Symptoms:
- chest pain (angina)
- shortness of breath
- fatigue
- palpitations
- nausea (sometimes)
Progression:
1. buildup of plaque in coronary arteries
2. narrowing of blood vessels
3. reduced oxygen supply to heart muscle
4. possible blockage leading to heart attack
Common Locations: coronary arteries, heart muscle
Duration: chronic (progressive over years)
Severity: moderate_to_severe
Complications: heart attack, heart failure, arrhythmias, sudden cardiac death
Home Remedy: lifestyle changes (healthy diet, exercise, smoking cessation, weight management)
Avoid: smoking, high-fat diet, sedentary lifestyle, unmanaged stress
When to See a Doctor: chest pain, shortness of breath, risk factors present
Emergency Signs: severe chest pain, pain radiating to arm or jaw, sweating, nausea, collapse
Prevention: healthy lifestyle, cholesterol control, blood pressure management, regular exercise
Contagious Period: none
Special Notes: leading cause of death globally, early detection and management reduce risk, may remain silent until severe
===
Disease: Heart Attack
Aliases: myocardial infarction, MI, cardiac infarction
Description: acute medical emergency caused by sudden blockage of blood flow to the heart muscle leading to tissue damage or death
Cause: rupture of atherosclerotic plaque with formation of a blood clot blocking a coronary artery
Transmission: none
Risk Groups: individuals with coronary artery disease, smokers, diabetics, hypertensive patients, obese individuals, elderly
Incubation Period: none
Symptoms:
- severe chest pain or pressure
- pain radiating to arm, jaw, neck, or back
- shortness of breath
- sweating
- nausea or vomiting
- dizziness or fainting
Progression:
1. plaque rupture in coronary artery
2. formation of blood clot
"""
DISEASE_TEXT_PART2 = r"""
Disease: Heart Attack
Aliases: myocardial infarction, MI, cardiac infarction
Description: acute medical emergency caused by sudden blockage of blood flow to the heart muscle leading to tissue damage or death
Cause: rupture of atherosclerotic plaque with formation of a blood clot blocking a coronary artery
Transmission: none
Risk Groups: individuals with coronary artery disease, smokers, diabetics, hypertensive patients, obese individuals, elderly
Incubation Period: none
Symptoms:
- severe chest pain or pressure
- pain radiating to arm, jaw, neck, or back
- shortness of breath
- sweating
- nausea or vomiting
- dizziness or fainting
Progression:
1. plaque rupture in coronary artery
2. formation of blood clot
3. blockage of blood flow to heart muscle
4. irreversible damage to heart tissue
Common Locations: heart muscle (myocardium), coronary arteries
Duration: acute (minutes to hours)
Severity: severe
Complications: heart failure, arrhythmias, cardiogenic shock, sudden death
Home Remedy: none
Avoid: delay in seeking treatment, smoking, unmanaged risk factors
When to See a Doctor: chest pain, shortness of breath, known heart disease symptoms
Emergency Signs: severe chest pain, collapse, difficulty breathing, loss of consciousness
Prevention: healthy lifestyle, managing cholesterol and blood pressure, quitting smoking, regular exercise
Contagious Period: none
Special Notes: requires immediate medical intervention, early treatment reduces damage, often preceded by angina
===
Disease: Stroke
Aliases: cerebrovascular accident, CVA, brain attack
Description: acute condition caused by interruption of blood flow to the brain leading to brain cell death and potential long-term disability or death
Cause: blockage of blood vessel (ischemic stroke) or rupture of blood vessel causing bleeding (hemorrhagic stroke)
Transmission: none
Risk Groups: individuals with hypertension, diabetes, heart disease, smokers, elderly, obese individuals, people with high cholesterol
Incubation Period: none
Symptoms:
- sudden numbness or weakness (especially one side of body)
- confusion or trouble speaking
- difficulty seeing in one or both eyes
- dizziness, loss of balance or coordination
- severe headache with no known cause
Progression:
1. interruption of blood supply to brain
2. brain cells deprived of oxygen and nutrients
3. rapid cell death within minutes
4. neurological deficits or permanent damage
Common Locations: brain, blood vessels supplying brain
Duration: acute event with long-term effects possible
Severity: severe
Complications: paralysis, speech impairment, cognitive deficits, disability, death
Home Remedy: none
Avoid: uncontrolled blood pressure, smoking, unhealthy lifestyle, delayed treatment
When to See a Doctor: any sudden neurological symptoms, suspected stroke
Emergency Signs: facial drooping, arm weakness, slurred speech, sudden severe headache, loss of consciousness
Prevention: healthy lifestyle, blood pressure control, cholesterol management, regular exercise, smoking cessation
Contagious Period: none
Special Notes: requires immediate emergency care, early treatment improves outcomes, TIA (mini-stroke) is a warning sign
===
Disease: Heart Failure
Aliases: congestive heart failure, CHF
Description: chronic condition where the heart cannot pump enough blood to meet the body's needs, leading to reduced organ function and fluid buildup
Cause: weakened or stiff heart muscle due to conditions like coronary artery disease, hypertension, heart attack, or cardiomyopathy
Transmission: none
Risk Groups: elderly, individuals with hypertension, coronary artery disease, diabetes, obesity, smokers, people with heart conditions
Incubation Period: none
Symptoms:
- shortness of breath
- fatigue and weakness
- swelling in legs, ankles, or abdomen
- persistent cough
- difficulty lying flat
- rapid weight gain from fluid buildup
Progression:
1. damage or weakening of heart muscle
2. reduced pumping ability of heart
3. blood flow slows and fluid accumulates
4. worsening symptoms and organ dysfunction
Common Locations: heart, lungs, blood vessels, kidneys
Duration: chronic (progressive over time)
Severity: moderate_to_severe
Complications: pulmonary edema, kidney failure, liver damage, arrhythmias, sudden cardiac arrest
Home Remedy: lifestyle changes (low-sodium diet, fluid control, rest, exercise as advised)
Avoid: excessive salt intake, smoking, alcohol misuse, untreated underlying conditions
When to See a Doctor: shortness of breath, swelling, fatigue, worsening symptoms
Emergency Signs: severe breathing difficulty, chest pain, confusion, sudden worsening of symptoms
Prevention: managing risk factors (blood pressure, diabetes), healthy lifestyle, regular medical care
Contagious Period: none
Special Notes: no cure but manageable with treatment, may worsen over time, early intervention improves quality of life
===
Disease: Atherosclerosis
Aliases: arterial plaque disease, hardening of the arteries
Description: chronic condition characterized by buildup of fatty plaques in arterial walls leading to narrowing and reduced blood flow
Cause: accumulation of cholesterol, fats, and inflammatory cells forming plaques in arteries
Transmission: none
Risk Groups: smokers, people with high cholesterol, hypertension, diabetes, obesity, sedentary lifestyle, older adults
Incubation Period: none
Symptoms:
- often asymptomatic early
- chest pain (if coronary arteries affected)
- leg pain while walking (claudication)
- fatigue
- shortness of breath
Progression:
1. endothelial damage in arteries
2. buildup of fatty deposits (plaques)
3. narrowing and stiffening of arteries
4. reduced blood flow or sudden blockage due to plaque rupture
Common Locations: arteries (coronary, carotid, peripheral)
Duration: chronic (progresses over years)
Severity: moderate_to_severe
Complications: heart attack, stroke, peripheral artery disease, organ damage
Home Remedy: lifestyle changes (healthy diet, exercise, weight management)
Avoid: smoking, high-fat diet, sedentary lifestyle, uncontrolled risk factors
When to See a Doctor: chest pain, leg pain, risk factors present, routine screening
Emergency Signs: sudden chest pain, stroke symptoms, severe limb pain, collapse
Prevention: healthy diet, regular exercise, cholesterol control, blood pressure management
Contagious Period: none
Special Notes: underlying cause of many cardiovascular diseases, often silent until complications occur
===
Disease: Peripheral Artery Disease
Aliases: PAD, peripheral vascular disease, PVD
Description: circulatory condition where narrowed peripheral arteries reduce blood flow to limbs, especially legs
Cause: atherosclerosis leading to plaque buildup and arterial narrowing
Transmission: none
Risk Groups: smokers, diabetics, elderly, individuals with high cholesterol or hypertension, sedentary lifestyle
Incubation Period: none
Symptoms:
- leg pain while walking (claudication)
- numbness or weakness in legs
- coldness in lower leg or foot
- sores on toes or feet that heal slowly
- hair loss or shiny skin on legs
Progression:
1. plaque buildup in peripheral arteries
2. narrowing reduces blood flow to limbs
3. symptoms appear during activity
4. worsening leads to rest pain and tissue damage
Common Locations: legs, feet, peripheral arteries
Duration: chronic (progressive over time)
Severity: moderate_to_severe
Complications: critical limb ischemia, ulcers, infections, amputation, increased risk of heart attack and stroke
Home Remedy: lifestyle changes (exercise, smoking cessation, healthy diet)
Avoid: smoking, inactivity, poor foot care, uncontrolled diabetes
When to See a Doctor: leg pain during walking, non-healing wounds, cold or numb limbs
Emergency Signs: severe leg pain at rest, ulcers, blackened tissue, signs of infection
Prevention: regular exercise, healthy diet, controlling risk factors (diabetes, cholesterol, blood pressure)
Contagious Period: none
Special Notes: often associated with atherosclerosis, early diagnosis improves outcomes, increases cardiovascular risk
===
Disease: Rheumatic Heart Disease
Aliases: RHD
Description: chronic heart condition caused by damage to heart valves following untreated or poorly treated rheumatic fever
Cause: autoimmune reaction following infection with Group A Streptococcus (streptococcal throat infection)
Transmission: not directly contagious (initial streptococcal infection spreads via respiratory droplets)
Risk Groups: children and adolescents, people in low-resource settings, individuals with untreated strep throat, overcrowded living conditions
Incubation Period: 2-4 weeks (after streptococcal infection leading to rheumatic fever)
Symptoms:
- shortness of breath
- chest pain
- fatigue
- heart palpitations
- swelling of legs or abdomen
- fainting (in severe cases)
Progression:
1. untreated streptococcal throat infection
2. development of rheumatic fever
3. autoimmune inflammation damages heart valves
4. chronic valve dysfunction leading to heart disease
Common Locations: heart valves (especially mitral and aortic valves)
Duration: chronic (lifelong condition)
Severity: moderate_to_severe
Complications: heart failure, valve stenosis or regurgitation, arrhythmias, stroke, death
Home Remedy: rest, adherence to medications, healthy lifestyle
Avoid: untreated strep throat, poor hygiene, missed antibiotic therapy
When to See a Doctor: sore throat with fever, symptoms of heart disease, known history of rheumatic fever
Emergency Signs: severe breathlessness, chest pain, fainting, signs of heart failure
Prevention: early treatment of streptococcal infections with antibiotics, good hygiene, long-term prophylactic antibiotics in high-risk individuals
Contagious Period: none (RHD itself not contagious)
Special Notes: preventable with timely treatment of strep throat, common in developing regions, requires long-term follow-up
===
Disease: Cardiomyopathy
Aliases: heart muscle disease
Description: group of diseases affecting the heart muscle leading to impaired pumping ability and structural abnormalities
Cause: genetic mutations, infections, alcohol abuse, toxins, hypertension, or unknown (idiopathic)
Transmission: none
Risk Groups: individuals with family history, alcohol or drug abuse, hypertension, viral infections, metabolic disorders
Incubation Period: none
Symptoms:
- shortness of breath
- fatigue
- swelling in legs and ankles
- chest pain
- palpitations
- dizziness or fainting
Progression:
1. damage or abnormality in heart muscle
2. reduced ability of heart to pump effectively
3. enlargement or stiffening of heart chambers
4. progression to heart failure or arrhythmias
Common Locations: heart muscle (myocardium)
Duration: chronic (may worsen over time)
Severity: moderate_to_severe
Complications: heart failure, arrhythmias, blood clots, sudden cardiac death
Home Remedy: lifestyle changes (healthy diet, limiting alcohol, rest, exercise as advised)
Avoid: alcohol abuse, smoking, untreated hypertension, strenuous activity if symptomatic
When to See a Doctor: shortness of breath, fatigue, chest pain, family history
Emergency Signs: severe chest pain, fainting, irregular heartbeat, sudden collapse
Prevention: managing risk factors, genetic counseling in familial cases, healthy lifestyle
Contagious Period: none
Special Notes: includes types like dilated, hypertrophic, and restrictive cardiomyopathy, may require medications or devices
===
Disease: Endocarditis
Aliases: infective endocarditis
Description: serious infection and inflammation of the inner lining of the heart chambers and valves (endocardium) caused by microorganisms
Cause: bacterial or fungal infection entering bloodstream and attaching to heart lining or valves
Transmission: bloodstream spread from sources like dental procedures, skin infections, intravenous drug use (not person-to-person)
Risk Groups: people with heart valve disease, congenital heart defects, prosthetic valves, IV drug users, immunocompromised individuals, poor dental hygiene
Incubation Period: days to weeks (variable)
Symptoms:
- fever and chills
- heart murmur
- chest pain
- fatigue
- shortness of breath
- night sweats
- skin spots or lesions
Progression:
1. microorganisms enter bloodstream
2. attach to heart lining or damaged valves
3. infection grows and forms vegetations
4. damage to heart valves and possible spread to other organs
Common Locations: heart valves, endocardium, bloodstream
Duration: weeks (requires prolonged treatment)
Severity: severe
Complications: heart valve damage, heart failure, embolism, stroke, sepsis, organ damage
Home Remedy: none
Avoid: poor dental hygiene, unsterile needle use, delayed treatment of infections
When to See a Doctor: persistent fever, fatigue, heart symptoms, risk factors present
Emergency Signs: severe chest pain, confusion, stroke symptoms, difficulty breathing
Prevention: good oral hygiene, prophylactic antibiotics for high-risk individuals, sterile medical practices
Contagious Period: none
Special Notes: requires urgent antibiotic or antifungal treatment, may need surgery, early diagnosis improves outcomes
===
Disease: Myocarditis
Aliases: inflammatory cardiomyopathy
Description: inflammation of the heart muscle that can reduce the heart's ability to pump and cause arrhythmias
Cause: viral infections (e.g., coxsackievirus, adenovirus), autoimmune reactions, bacterial or fungal infections, toxins, certain medications
Transmission: none (underlying infections may be contagious depending on cause)
Risk Groups: young adults, children, individuals with viral infections, immunocompromised individuals, people exposed to toxins or certain drugs
Incubation Period: variable (depends on underlying cause)
Symptoms:
- chest pain
- shortness of breath
- fatigue
- palpitations
- fever
- swelling in legs (in severe cases)
Progression:
1. infection or immune trigger affects heart muscle
2. inflammation of myocardium
3. impaired heart function and electrical activity
4. recovery or progression to heart failure or chronic damage
Common Locations: heart muscle (myocardium)
Duration: days to months (may become chronic)
Severity: mild_to_severe
Complications: heart failure, arrhythmias, cardiomyopathy, sudden cardiac death
Home Remedy: rest, avoid strenuous activity, supportive care
Avoid: intense physical activity during illness, alcohol, delayed treatment
When to See a Doctor: chest pain, shortness of breath, palpitations, recent infection
Emergency Signs: severe chest pain, fainting, irregular heartbeat, difficulty breathing
Prevention: infection control, vaccination where applicable, avoiding toxins
Contagious Period: none
Special Notes: often follows viral infection, may resolve completely or lead to chronic heart disease, early rest is important
===
Disease: Pericarditis
Aliases: pericardial inflammation
Description: inflammation of the pericardium (the fluid-filled sac surrounding the heart) causing chest pain and possible fluid accumulation
Cause: viral infections (most common), bacterial infections, autoimmune disorders, post-heart attack inflammation, trauma
Transmission: none (underlying infections may be contagious depending on cause)
Risk Groups: individuals with recent viral infections, autoimmune diseases, post-cardiac injury, immunocompromised individuals
Incubation Period: none
Symptoms:
- sharp chest pain (worsens with breathing or lying down)
- fever
- fatigue
- shortness of breath
- dry cough
Progression:
1. inflammation of pericardium
2. accumulation of fluid around heart (pericardial effusion)
3. increased pressure on heart
4. resolution or progression to complications
Common Locations: pericardium (outer lining of heart)
Duration: days to weeks (may recur)
Severity: mild_to_moderate
Complications: pericardial effusion, cardiac tamponade, chronic or recurrent pericarditis
Home Remedy: rest, anti-inflammatory measures, hydration
Avoid: strenuous activity, delayed treatment, ignoring symptoms
When to See a Doctor: chest pain, fever, recent infection, breathing discomfort
Emergency Signs: severe chest pain, difficulty breathing, fainting, signs of cardiac tamponade
Prevention: treating underlying infections, managing autoimmune conditions
Contagious Period: none
Special Notes: chest pain often relieved by sitting forward, usually self-limiting but requires monitoring
===
Disease: Varicose Veins
Aliases: varicosities
Description: enlarged, twisted veins caused by valve dysfunction leading to poor blood flow, commonly affecting the legs
Cause: weakened or damaged vein valves causing blood pooling and vein dilation
Transmission: none
Risk Groups: older adults, women, pregnant individuals, people with prolonged standing, obesity, family history
Incubation Period: none
Symptoms:
- visible twisted, bulging veins
- aching or heavy legs
- swelling in lower legs
- itching around veins
- muscle cramps
Progression:
1. weakening of vein valves
2. blood pools in veins
3. veins enlarge and become visible
4. worsening discomfort and complications
Common Locations: legs, feet
Duration: chronic (progressive over time)
Severity: mild_to_moderate
Complications: skin ulcers, bleeding, thrombophlebitis, chronic venous insufficiency
Home Remedy: leg elevation, exercise, compression stockings, weight management
Avoid: prolonged standing or sitting, tight clothing, inactivity
When to See a Doctor: pain, swelling, skin changes, bleeding veins
Emergency Signs: severe swelling, redness, pain suggesting clot, bleeding
Prevention: regular exercise, weight control, avoiding long periods of standing, leg elevation
Contagious Period: none
Special Notes: common condition, cosmetic and medical concerns, treatments include sclerotherapy and surgery
===
Disease: Deep Vein Thrombosis
Aliases: DVT
Description: condition where a blood clot forms in a deep vein, usually in the legs, potentially leading to life-threatening complications
Cause: blood clot formation due to reduced blood flow, vessel injury, or hypercoagulability (Virchow's triad)
Transmission: none
Risk Groups: immobile individuals, recent surgery patients, pregnant women, smokers, obese individuals, people with clotting disorders
Incubation Period: none
Symptoms:
- swelling in one leg
- pain or tenderness in leg
- warmth in affected area
- redness or discoloration
- visible surface veins
Progression:
1. formation of clot in deep vein
2. obstruction of blood flow
3. local inflammation and swelling
4. possible clot dislodgement leading to embolism
Common Locations: deep veins of legs (calf, thigh)
Duration: acute to chronic (depends on treatment)
Severity: moderate_to_severe
Complications: pulmonary embolism, post-thrombotic syndrome, chronic venous insufficiency
Home Remedy: none
Avoid: prolonged immobility, dehydration, smoking, delayed treatment
When to See a Doctor: leg swelling, pain, redness, risk factors present
Emergency Signs: sudden shortness of breath, chest pain, coughing blood (possible pulmonary embolism)
Prevention: mobility, compression stockings, anticoagulants in high-risk individuals, hydration
Contagious Period: none
Special Notes: early diagnosis critical, may be silent, requires anticoagulant therapy
===
Disease: Chronic Obstructive Pulmonary Disease
Aliases: COPD, chronic bronchitis, emphysema
Description: progressive lung disease causing airflow limitation and breathing difficulty due to chronic inflammation and damage to airways
Cause: long-term exposure to irritants such as cigarette smoke, air pollution, occupational dusts, genetic factors (e.g., alpha-1 antitrypsin deficiency)
Transmission: none
Risk Groups: smokers, long-term exposure to air pollutants, older adults, occupational exposure to dust and chemicals
Incubation Period: none
Symptoms:
- chronic cough
- shortness of breath
- wheezing
- chest tightness
- excessive mucus production
- fatigue
Progression:
1. prolonged exposure to lung irritants
2. inflammation and damage to airways and alveoli
3. airflow limitation and reduced oxygen exchange
4. progressive worsening of respiratory function
Common Locations: lungs, airways
Duration: chronic (lifelong, progressive)
Severity: moderate_to_severe
Complications: respiratory failure, pulmonary hypertension, heart problems (cor pulmonale), frequent infections
Home Remedy: smoking cessation, pulmonary rehabilitation, breathing exercises, proper nutrition
Avoid: smoking, air pollutants, respiratory infections, poor adherence to treatment
When to See a Doctor: persistent cough, breathlessness, worsening symptoms
Emergency Signs: severe breathlessness, bluish lips, confusion, inability to speak full sentences
Prevention: avoid smoking, reduce exposure to pollutants, vaccinations (flu, pneumococcal), protective measures
Contagious Period: none
Special Notes: irreversible but manageable, early diagnosis improves quality of life, exacerbations can be life-threatening
===
Disease: Bronchitis
Aliases: acute bronchitis, chronic bronchitis
Description: inflammation of the bronchial tubes causing cough and mucus production, can be acute or part of chronic lung disease
Cause: viral infections (most common), bacterial infections, or long-term exposure to irritants like smoke and pollution
Transmission: respiratory droplets (in infectious cases), close contact with infected individuals
Risk Groups: smokers, elderly, children, people with weakened immune systems, individuals exposed to pollutants
Incubation Period: 1-4 days (for viral bronchitis)
Symptoms:
- persistent cough (often with mucus)
- chest discomfort
- fatigue
- shortness of breath
- mild fever and chills
- wheezing
Progression:
1. infection or irritation of bronchial tubes
2. inflammation and mucus production
3. persistent cough develops
4. recovery (acute) or progression to chronic condition
Common Locations: bronchial tubes, lungs
Duration: acute (1-3 weeks), chronic (months to years)
Severity: mild_to_moderate
Complications: pneumonia, chronic bronchitis, worsening of COPD, respiratory distress
Home Remedy: rest, hydration, warm fluids, humidified air
Avoid: smoking, air pollutants, cold exposure, untreated infections
When to See a Doctor: prolonged cough (>3 weeks), high fever, breathing difficulty, blood in mucus
Emergency Signs: severe breathlessness, chest pain, bluish lips, confusion
Prevention: hand hygiene, avoiding smoking, vaccinations, minimizing exposure to irritants
Contagious Period: during active infection (especially viral stage)
Special Notes: acute bronchitis is usually viral and self-limiting, antibiotics often not needed, chronic bronchitis is a form of COPD
===
Disease: Pneumonia
Aliases: lung infection
Description: infection of the lungs causing inflammation of air sacs (alveoli) which may fill with fluid or pus, impairing breathing
Cause: bacteria (e.g., Streptococcus pneumoniae), viruses, fungi, or aspiration of foreign material
Transmission: respiratory droplets, inhalation of infectious agents, aspiration
Risk Groups: infants, elderly, smokers, immunocompromised individuals, people with chronic diseases
Incubation Period: 1-4 days (varies by cause)
Symptoms:
- cough (with or without mucus)
- fever and chills
- shortness of breath
- chest pain (worsens with breathing)
- fatigue
- confusion (especially in elderly)
Progression:
1. infection reaches lungs
2. inflammation of alveoli
3. fluid or pus accumulation
4. impaired oxygen exchange
Common Locations: lungs (alveoli)
Duration: 1-3 weeks (longer in severe cases)
Severity: moderate_to_severe
Complications: respiratory failure, sepsis, pleural effusion, lung abscess, death
Home Remedy: rest, hydration, fever control, supportive care
Avoid: smoking, delayed treatment, exposure to infections
When to See a Doctor: high fever, breathing difficulty, chest pain, persistent cough
Emergency Signs: severe breathlessness, bluish lips, confusion, high fever, low oxygen levels
Prevention: vaccination (pneumococcal, flu), hand hygiene, avoiding smoking, good nutrition
Contagious Period: varies depending on cause (bacterial or viral)
Special Notes: early treatment improves outcomes, high-risk groups need prompt care, can be life-threatening
===
Disease: Lung Cancer
Aliases: pulmonary cancer
Description: malignant tumor of lung tissue characterized by uncontrolled cell growth, often associated with smoking and environmental exposures
Cause: genetic mutations in lung cells triggered by smoking, air pollution, radon gas, occupational exposures (e.g., asbestos)
Transmission: none
Risk Groups: smokers, passive smokers, elderly, individuals exposed to carcinogens, people with family history of cancer
Incubation Period: none
Symptoms:
- persistent cough
- coughing up blood (hemoptysis)
- chest pain
- shortness of breath
- weight loss
- fatigue
Progression:
1. genetic mutations in lung cells
2. uncontrolled cell growth forming tumors
3. invasion of surrounding tissues
4. metastasis to other organs
Common Locations: lungs, may spread to brain, liver, bones
Duration: chronic (progressive over time)
Severity: severe
Complications: metastasis, respiratory failure, pleural effusion, death
Home Remedy: none
Avoid: smoking, exposure to carcinogens, air pollution
When to See a Doctor: persistent cough, unexplained weight loss, chest pain, coughing blood
Emergency Signs: severe breathing difficulty, massive hemoptysis, chest pain, collapse
Prevention: smoking cessation, avoiding pollutants, early screening in high-risk individuals
Contagious Period: none
Special Notes: leading cause of cancer-related deaths, early detection improves survival, includes types like small cell and non-small cell lung cancer
===
Disease: Sinusitis
Aliases: sinus infection, rhinosinusitis
Description: inflammation of the sinuses causing blockage and buildup of mucus, often due to infection or allergies
Cause: viral infections (most common), bacterial or fungal infections, allergies, nasal polyps
Transmission: respiratory droplets (in infectious cases), not contagious in non-infectious cases
Risk Groups: people with allergies, asthma, frequent colds, smokers, individuals with weak immune systems
Incubation Period: 1-3 days (for viral causes)
Symptoms:
- facial pain or pressure
- nasal congestion
- thick nasal discharge
- headache
- reduced sense of smell
- fever (in some cases)
Progression:
1. inflammation of sinus lining
2. blockage of sinus drainage
3. mucus accumulation
4. infection or prolonged inflammation
Common Locations: paranasal sinuses (frontal, maxillary, ethmoid, sphenoid)
Duration: acute (up to 4 weeks), subacute (4-12 weeks), chronic (more than 12 weeks)
Severity: mild_to_moderate
Complications: chronic sinusitis, spread of infection to eyes or brain (rare), abscess
Home Remedy: steam inhalation, hydration, saline nasal irrigation, rest
Avoid: allergens, smoking, untreated infections, poor hygiene
When to See a Doctor: symptoms lasting more than 10 days, severe pain, high fever, recurrent infections
Emergency Signs: vision changes, severe headache, swelling around eyes, confusion
Prevention: hand hygiene, allergy control, avoiding irritants, vaccination
Contagious Period: varies (only if viral or bacterial cause)
Special Notes: most cases are viral and resolve without antibiotics, chronic cases may require long-term management
===
Disease: Pulmonary Fibrosis
Aliases: interstitial lung disease (fibrotic type), lung fibrosis
Description: chronic progressive lung disease characterized by scarring of lung tissue leading to reduced oxygen exchange
Cause: idiopathic (unknown) or due to long-term exposure to toxins, autoimmune diseases, infections, medications, or radiation
Transmission: none
Risk Groups: older adults, smokers, people exposed to dusts (asbestos, silica), individuals with autoimmune disorders, family history
Incubation Period: none
Symptoms:
- shortness of breath (especially on exertion)
- dry persistent cough
- fatigue
- unexplained weight loss
- clubbing of fingers (in advanced cases)
Progression:
1. injury to lung tissue
2. abnormal healing and fibrosis (scarring)
3. stiffening of lungs
4. progressive decline in lung function
Common Locations: lung interstitium (alveolar walls)
Duration: chronic (progressive over years)
Severity: moderate_to_severe
Complications: respiratory failure, pulmonary hypertension, heart failure (cor pulmonale), death
Home Remedy: oxygen therapy support, pulmonary rehabilitation, nutrition support
Avoid: smoking, exposure to dust and toxins, delayed treatment
When to See a Doctor: persistent cough, breathlessness, fatigue, risk factors present
Emergency Signs: severe breathlessness, low oxygen levels, cyanosis, confusion
Prevention: avoiding environmental exposures, managing underlying conditions, early detection
Contagious Period: none
Special Notes: often irreversible, progression varies, antifibrotic drugs may slow disease
===
Disease: Pleural Effusion
Aliases: fluid in pleural space
Description: accumulation of excess fluid between the layers of the pleura surrounding the lungs, impairing breathing
Cause: heart failure, infections (pneumonia, tuberculosis), cancer, liver or kidney disease, pulmonary embolism
Transmission: none
Risk Groups: individuals with heart disease, lung infections, cancer, chronic liver or kidney disease, elderly
Incubation Period: none
Symptoms:
- shortness of breath
- chest pain (worsens with breathing)
- dry cough
- fatigue
- reduced breath sounds
Progression:
1. underlying disease causes fluid buildup
2. accumulation in pleural space
3. compression of lung tissue
4. impaired breathing and oxygen exchange
Common Locations: pleural space (around lungs)
Duration: variable (depends on underlying cause)
Severity: mild_to_severe
Complications: respiratory distress, infection (empyema), lung collapse, chronic effusion
Home Remedy: none
Avoid: untreated underlying conditions, delayed diagnosis
When to See a Doctor: breathlessness, chest pain, persistent cough, known risk factors
Emergency Signs: severe breathing difficulty, low oxygen levels, chest tightness, rapid worsening
Prevention: managing underlying diseases, early treatment of infections, regular health monitoring
Contagious Period: none
Special Notes: treatment depends on cause, may require drainage, can recur if underlying condition persists
===
Disease: Pneumothorax
Aliases: collapsed lung
Description: condition where air enters the pleural space causing partial or complete collapse of the lung
Cause: rupture of lung blebs, chest injury, lung disease (e.g., COPD), medical procedures, spontaneous causes
Transmission: none
Risk Groups: tall thin individuals, smokers, people with lung disease, trauma patients, individuals undergoing chest procedures
Incubation Period: none
Symptoms:
- sudden chest pain
- shortness of breath
- rapid breathing
- decreased breath sounds on affected side
- fatigue
Progression:
1. air leaks into pleural space
2. increased pressure compresses lung
3. partial or complete lung collapse
4. worsening respiratory distress if untreated
Common Locations: pleural space, lungs
Duration: acute (may resolve or require intervention)
Severity: moderate_to_severe
Complications: tension pneumothorax, respiratory failure, recurrence
Home Remedy: none
Avoid: smoking, high-altitude exposure (if prone), untreated lung disease
When to See a Doctor: sudden chest pain, breathing difficulty
Emergency Signs: severe breathlessness, chest tightness, cyanosis, low blood pressure (tension pneumothorax)
Prevention: smoking cessation, managing lung diseases, avoiding risk factors
Contagious Period: none
Special Notes: tension pneumothorax is life-threatening emergency, may require needle decompression or chest tube
===
Disease: Sleep Apnea
Aliases: obstructive sleep apnea, OSA, central sleep apnea, CSA
Description: sleep disorder in which breathing repeatedly stops and starts during sleep, leading to poor oxygenation and disrupted rest
Cause: obstructive (relaxation of throat muscles blocking airway), central (brain fails to send proper signals to breathing muscles), or mixed
Transmission: none
Risk Groups: overweight individuals, older adults, males, people with large neck circumference, alcohol users, smokers, those with nasal obstruction or family history
Incubation Period: none
Symptoms:
- loud snoring
- pauses in breathing during sleep (noticed by others)
- gasping or choking during sleep
- excessive daytime sleepiness
- morning headache
- difficulty concentrating
Progression:
1. airway obstruction or breathing signal failure during sleep
2. repeated pauses in breathing (apnea episodes)
3. oxygen levels drop and sleep is disrupted
4. chronic fatigue and long-term complications develop
Common Locations: upper airway, brain (in central type)
Duration: chronic (long-term condition)
Severity: mild_to_severe
Complications: hypertension, heart disease, stroke, diabetes, daytime fatigue, accidents
Home Remedy: weight loss, sleeping on side, avoiding alcohol and sedatives, maintaining sleep routine
Avoid: alcohol, smoking, sedatives, sleeping on back (in OSA)
When to See a Doctor: loud snoring, daytime sleepiness, witnessed breathing pauses
Emergency Signs: severe breathing interruptions, extreme daytime drowsiness affecting safety, chest pain
Prevention: healthy weight, regular exercise, avoiding alcohol/sedatives, good sleep hygiene
Contagious Period: none
Special Notes: commonly treated with CPAP (continuous positive airway pressure) therapy, often underdiagnosed
===
Disease: Sarcoidosis
Aliases: none
Description: inflammatory disease characterized by formation of granulomas (tiny clumps of inflammatory cells) in various organs, most commonly the lungs and lymph nodes
Cause: unknown (likely abnormal immune response to environmental or infectious triggers in genetically predisposed individuals)
Transmission: none
Risk Groups: adults (20-40 years), people with family history, certain ethnic groups, individuals exposed to environmental triggers
Incubation Period: none
Symptoms:
- persistent dry cough
- shortness of breath
- chest pain
- fatigue
- fever
- skin rashes or nodules
- swollen lymph nodes
Progression:
1. immune system overreacts to unknown trigger
2. formation of granulomas in affected organs
3. inflammation disrupts normal organ function
4. may resolve or progress to chronic fibrosis
Common Locations: lungs, lymph nodes, skin, eyes, liver
Duration: acute (months) or chronic (years)
Severity: mild_to_severe
Complications: pulmonary fibrosis, vision problems, heart involvement (arrhythmias), organ dysfunction
Home Remedy: rest, healthy lifestyle, avoiding environmental triggers
Avoid: exposure to dust, chemicals, smoking, delayed treatment
When to See a Doctor: persistent cough, breathing difficulty, unexplained fatigue or swelling
Emergency Signs: severe breathlessness, chest pain, vision loss, irregular heartbeat
Prevention: no known prevention (cause unclear)
Contagious Period: none
Special Notes: many cases resolve spontaneously, others require corticosteroids or immunosuppressive therapy
===
Disease: Occupational Lung Disease
Aliases: work-related lung disease, pneumoconiosis
Description: group of lung conditions caused by long-term inhalation of harmful substances at the workplace leading to lung damage and breathing problems
Cause: exposure to dust (silica, coal, asbestos), chemicals, fumes, gases, or allergens in occupational settings
Transmission: none
Risk Groups: miners, construction workers, factory workers, farmers, textile workers, industrial laborers, people exposed to workplace pollutants
Incubation Period: none (develops over months to years of exposure)
Symptoms:
- chronic cough
- shortness of breath
- chest tightness
- wheezing
- fatigue
Progression:
1. inhalation of harmful particles or chemicals
2. accumulation and irritation in lungs
3. inflammation and lung tissue damage
4. fibrosis or chronic respiratory disease
Common Locations: lungs, airways
Duration: chronic (progressive with continued exposure)
Severity: mild_to_severe
Complications: pulmonary fibrosis, COPD, lung cancer, respiratory failure
Home Remedy: avoid exposure, use protective equipment, healthy lifestyle
Avoid: continued exposure to harmful substances, smoking, ignoring safety measures
When to See a Doctor: persistent cough, breathing difficulty, known workplace exposure
Emergency Signs: severe breathlessness, chest pain, oxygen deficiency, confusion
Prevention: workplace safety measures, masks/respirators, proper ventilation, regular health checkups
Contagious Period: none
Special Notes: includes diseases like asbestosis, silicosis, and coal workers' pneumoconiosis, prevention is key
===
Disease: Depression
Aliases: clinical depression, dysthymic disorder, major depressive disorder, unipolar depression, seasonal affective disorder, bipolar depression, psychotic depression
Description: serious mood disorder that affects how you think and feel, causing continuous sadness, low mood, and loss of interest in day-to-day activities
Cause: complex combination of genetic, biological, environmental, and psychological factors, or stressful life events
Transmission: none
Risk Groups: women, individuals with a family history of depression, people experiencing trauma or stressful life events, those with other mental disorders or chronic medical conditions
Incubation Period: none
Symptoms:
- continuous sadness, low mood, or feeling empty
- irritable, angry, or frustrated over small things
- guilty, hopeless, or tearful
- tired, low energy, or speaking and moving slower than usual
- negative self-worth or low confidence
- difficulty concentrating, remembering things, or making decisions
- sleeping too much or having problems sleeping
- losing or gaining appetite and weight
- unexplained aches, pains, or digestive problems
- thoughts of self-harm, death, or suicide
Progression:
1. symptoms start and progress gradually
2. daily activities like sleeping, eating, or working become increasingly difficult
3. individuals may isolate themselves and avoid social activities or hobbies
4. left untreated, symptoms persist and can lead to severe episodes, psychosis, or suicidal thoughts
Common Locations: none
Duration: long-term (episodes last at least two weeks, persistent depressive disorder lasts at least two years)
Severity: mild_to_severe
Complications: increased risk of other medical conditions, suicidal thoughts, strained relationships, difficulties at work or home, psychosis
Home Remedy: regular exercise, consistent sleep schedule, controlling stress, talking to trusted people, creative outlets like art or journaling
Avoid: increasing use of alcohol, tobacco, or drugs, isolating from family and friends
When to See a Doctor: depression affects daily life, finding it hard to manage responsibilities, not looking after yourself, symptoms last at least two weeks
Emergency Signs: thoughts of self-harm, suicidal thoughts, hallucinations, delusions
Prevention: mostly cannot be prevented, but healthy lifestyle changes like regular exercise and consistent sleep can help manage mental health
Contagious Period: none
Special Notes: psychotic depression involves hallucinations or delusions, treatments include cognitive behavioral therapy (CBT), antidepressants, electroconvulsive therapy (ECT), and repetitive transcranial magnetic stimulation (rTMS)
===
Disease: Schizophrenia
Aliases: none
Description: serious mental health condition and psychotic illness that distorts a person's thinking, causing them to lose touch with reality
Cause: combination of genetics, differences in brain structure and chemicals, pregnancy or birth complications, and trauma
Transmission: none
Risk Groups: starts between ages 16 and 30, men (develop symptoms earlier), people with a family history, individuals who experienced childhood trauma, abuse, or neglect
Incubation Period: none
Symptoms:
- hallucinations (hearing voices or seeing things that are not there)
- delusions (unusual false beliefs not based on reality)
- muddled thoughts, disorganized speech, and trouble using information
- lack of interest in things and difficulty showing emotions
- strange movements, psychomotor agitation, or psychomotor retardation
- difficulty concentrating, paying attention, or making decisions
- feeling fearful, worried, or suspicious that others wish them harm
Progression:
1. symptoms may start suddenly or develop slowly over time
2. positive (psychotic), negative, and cognitive symptoms emerge
3. episodes of severe symptoms occur, sometimes followed by periods of few or no symptoms
4. functioning in daily life, keeping a job, and self-care become highly difficult
Common Locations: none
Duration: lifelong (though about one third recover completely after one episode)
Severity: severe
Complications: inability to work or care for oneself, severe relapse, suicidal thoughts
Home Remedy: establish a regular routine, get plenty of sleep, eat a healthy diet, exercise regularly
Avoid: smoking, recreational drugs, drinking too much alcohol, severe stress
When to See a Doctor: changes in thoughts or behavior that last a long time, current treatments are not helping, warning signs like hearing quiet voices or feeling suspicious appear
Emergency Signs: mental health crisis, severe psychotic episode, refusing care leading to potential harm
Prevention: chance of severe relapse reduced if well-managed, maintain healthy lifestyle and routines
Contagious Period: none
Special Notes: treated with antipsychotic medication and psychological therapies (CBT, family therapy), advanced statements can be written to record care preferences in case of a serious episode
===
Disease: Obsessive-Compulsive Disorder
Aliases: OCD
Description: mental disorder characterized by repeated, unwanted thoughts (obsessions) and irrational, repetitive behaviors (compulsions) performed to reduce anxiety
Cause: unknown, likely a combination of genetics, brain biology and chemistry, and environmental factors
Transmission: none
Risk Groups: teens, young adults, boys (develop earlier than girls), people with a first-degree relative with OCD, individuals with childhood trauma
Incubation Period: none
Symptoms:
- repeated unwanted thoughts, urges, or mental images causing anxiety
- intense fear of germs, contamination, or losing something
- aggressive or forbidden thoughts involving sex or religion
- excessive cleaning, handwashing, or ordering things precisely
- repeatedly checking things (e.g., locks, ovens) or compulsive counting
- sudden twitches, movements, or sounds (if co-occurring with a tic disorder)
Progression:
1. intrusive, distressing thought or urge enters the mind (obsession)
2. obsession causes a feeling of intense anxiety or distress
3. repetitive behavior or mental act is performed to cope (compulsion)
4. compulsion brings temporary relief until the anxiety returns and the cycle repeats
Common Locations: none
Duration: lifelong chronic condition
Severity: moderate_to_severe
Complications: depression, eating disorders, generalized anxiety disorder, hoarding disorder, Tourette syndrome, suicidal feelings
Home Remedy: utilize self-help guides for mild to moderate symptoms, participate in online or local support groups
Avoid: giving in to compulsions without seeking therapy
When to See a Doctor: obsessive thoughts and compulsive behaviors affect daily life, unable to manage responsibilities, spending at least 1 hour a day on these thoughts or behaviors
Emergency Signs: suicidal feelings
Prevention: none
Contagious Period: none
Special Notes: children may develop OCD symptoms following a streptococcal infection (PANDAS), primary treatments include Cognitive Behavioral Therapy (specifically Exposure and Response Prevention) and antidepressants
===
Disease: Post-Traumatic Stress Disorder
Aliases: PTSD, CPTSD, complex PTSD
Description: mental health problem developed after experiencing or witnessing a life-threatening or traumatic event, causing prolonged stress, fear, and flashbacks
Cause: severe trauma such as road accidents, assault, abuse, combat, natural disasters, or the sudden death of a loved one
Transmission: none
Risk Groups: women, people with childhood trauma, individuals lacking social support, people with a history of mental illness or substance abuse, emergency service workers, military personnel
Incubation Period: symptoms usually start soon after the event but can appear months or years later
Symptoms:
- nightmares, flashbacks, and intrusive frightening thoughts
- physical sensations like pain, sweating, trembling, or feeling sick
- feeling alert, on edge, easily startled, or jumpy (hypervigilance)
- avoiding people, places, or talking about the trauma
- feeling numb, detached from body, or unable to show affection
- negative thoughts about self, guilt, blame, or trust issues
- anger, irritability, and trouble sleeping or concentrating
Progression:
1. person experiences or witnesses a traumatic event
2. fight-or-flight response triggers intense fear and stress
3. fear and stress fail to subside naturally over time
4. re-experiencing, avoidance, arousal, and negative mood symptoms persist and interfere with daily life
Common Locations: none
Duration: long-term (symptoms must last longer than four weeks for diagnosis, can come and go over years)
Severity: moderate_to_severe
Complications: depression, substance use, sleep problems, chronic pain
Home Remedy: seek support from friends and family, learn coping strategies, join a support group
Avoid: using alcohol or drugs to cope, isolating oneself, avoiding thoughts or feelings completely
When to See a Doctor: symptoms last longer than 4 weeks after the trauma, symptoms cause great distress or interfere with work and home life
Emergency Signs: severe self-destructive or reckless behavior
Prevention: seek support early, develop coping strategies, learn to respond effectively to fear (resilience factors)
Contagious Period: none
Special Notes: treatments include trauma-focused cognitive behavioral therapy (CBT), Eye Movement Desensitization and Reprocessing (EMDR), and sometimes antidepressants
===
Disease: Panic Disorder
Aliases: none
Description: type of anxiety disorder causing repeated, sudden periods of intense fear or discomfort (panic attacks) when there is no real danger
Cause: unknown, likely a combination of genetics, brain biology, environment, and major stress
Transmission: none
Risk Groups: women, late teens or early adults, people under a lot of stress, individuals with a history of childhood trauma
Incubation Period: none
Symptoms:
- sudden, repeated panic attacks of overwhelming anxiety and fear
- pounding or racing heart, chest pain
- sweating, chills, trembling, or shaking
- trouble breathing or the feeling of choking
- weakness, dizziness, stomach pain, or nausea
- feeling out of control or intense fear of death
- avoiding places where past attacks occurred
Progression:
1. individual experiences a sudden panic attack without warning
2. physical symptoms like racing heart and shortness of breath peak within minutes
3. individual develops an intense worry about having another attack
4. individual begins avoiding places or situations associated with past attacks
Common Locations: none
Duration: long-term, individual attacks last a few minutes to over an hour
Severity: moderate_to_severe
Complications: depression, substance use disorders, agoraphobia
Home Remedy: eat regular meals, get enough sleep, exercise regularly, join a support group
Avoid: alcohol, caffeine
When to See a Doctor: experiencing repeated panic attacks, anxiety about future attacks interferes with quality of life
Emergency Signs: severe chest pain or trouble breathing that mimics a heart attack
Prevention: healthy lifestyle choices, stress management
Contagious Period: none
Special Notes: treated with talk therapy (CBT), antidepressants (SSRIs, SNRIs), and anti-anxiety medicines
===
Disease: Eating Disorders
Aliases: anorexia nervosa, AN, binge eating disorder, BED, bulimia nervosa, BN, avoidant restrictive food intake disorder, ARFID, OSFED
Description: serious mental health conditions involving severe problems with thoughts about food, body shape, and irregular eating behaviors
Cause: complex interaction of genetic, biological, behavioral, psychological, and social factors
Transmission: none
Risk Groups: women, teenagers, young adults, individuals with other mental health conditions like anxiety, depression, or OCD
Incubation Period: none
Symptoms:
- eating unusually large amounts of food or very little food
- intensive and excessive exercise, purging (vomiting/laxatives), or fasting
- distorted body image or intense fear of gaining weight
- feeling dizzy, tired, cold, faint, or weak
- digestive problems, bloating, severe constipation, or GERD
- thinning bones, brittle hair/nails, dry skin, or tooth decay
- missing periods, delayed puberty, or infertility
Progression:
1. unhealthy thoughts about weight, body shape, and food develop
2. individual adopts extreme eating behaviors (restricting, binging, purging)
3. physical health begins to deteriorate due to poor nutrition and electrolyte imbalances
4. untreated conditions lead to severe complications, multiorgan failure, or death
Common Locations: systemic
Duration: long-term
Severity: moderate_to_severe
Complications: heart and kidney problems, bone thinning, severe dehydration, multiorgan failure, infertility, brain damage, death
Home Remedy: self-help or guided self-help for mild cases, developing healthy eating habits
Avoid: extreme diets, isolating from friends and family, intense focus on body weight and shape
When to See a Doctor: warning signs like rapid weight changes, secret eating, purging, or obsessive exercising are noticed
Emergency Signs: fainting, severe dehydration, irregular heartbeat, multiorgan failure, suicidal thoughts
Prevention: promote healthy body image, recognize early warning signs, encourage open communication about feelings
Contagious Period: none
Special Notes: anorexia nervosa has the highest death rate of any mental disorder, treatment includes psychotherapy (CBT), nutrition counseling, medical monitoring, and sometimes antidepressants or mood stabilizers
===
Disease: Drug Use and Addiction
Aliases: drug abuse, substance use, substance use disorder
Description: chronic, relapsing brain disease causing repeated, compulsive drug seeking and use despite harmful physical and psychological consequences
Cause: repeated use of drugs altering brain structure and function, combined with genetic, environmental, and developmental factors
Transmission: none
Risk Groups: people with untreated mental health problems, individuals with childhood trauma or unhappy home environments, youth who start using early, people exposed to peer pressure
Incubation Period: none
Symptoms:
- taking larger doses than prescribed or using drugs for non-medical reasons
- changing friends, spending time alone, or losing interest in hobbies
- poor hygiene and neglect of personal care
- erratic behavior, rapid mood swings, being very energetic, or nonsensical speech
- changes in eating or sleeping habits
- missing appointments and struggling at work, school, or in relationships
Progression:
1. initial drug use alters brain chemistry and creates a high or relief
2. tolerance builds, requiring more of the substance to achieve the same effect
3. brain changes make drug-seeking behavior compulsive despite negative consequences
4. addiction takes hold, leading to severe social, physical, and mental health decline, with risk of relapse even after quitting
Common Locations: brain, systemic
Duration: long-term (chronic and relapsing)
Severity: moderate_to_severe
Complications: permanent brain and body damage, harm to fetus if pregnant, infectious diseases (like HIV), overdose, death
Home Remedy: seek strong family and friend support systems, practice stress management
Avoid: people, places, and situations where drugs are accessible
When to See a Doctor: drug use interferes with daily life, experiencing withdrawal symptoms when trying to quit, signs of addiction or dependence appear
Emergency Signs: overdose, severe withdrawal symptoms, extreme erratic or dangerous behavior
Prevention: education on drug risks, stable home environments, early intervention, treating underlying mental health issues
Contagious Period: none
Special Notes: combining medicines with counseling gives the best chance of success, and dual diagnosis (treating co-occurring mental disorders) is crucial for recovery
===
Disease: Osteoarthritis
Aliases: degenerative joint disease, OA, wear-and-tear arthritis
Description: chronic degenerative joint disorder caused by progressive breakdown of cartilage and changes in bone, leading to pain, stiffness, reduced movement, and functional limitation
Cause: multifactorial, age-related cartilage degeneration, joint mechanical stress, prior injury, obesity, genetics, abnormal joint structure
Transmission: none
Risk Groups: older adults, women especially after age 50, overweight or obese individuals, prior joint injury or surgery, repetitive joint overuse, family history, congenital joint abnormalities
Incubation Period: none
Symptoms:
- joint pain worsened by activity, relieved by rest in early stages
- stiffness after rest or waking, usually brief morning stiffness
- swelling, tenderness, reduced range of motion, weakness around joint
- grinding, crunching, crepitus, feeling of bone rubbing
- bony enlargements of fingers, enlarged or gnarled joints
- difficulty walking, climbing stairs, gripping objects, dressing, standing from chairs
- severe persistent pain, joint instability, deformity, disability in advanced disease
Progression:
1. cartilage gradually thins and loses cushioning ability
2. intermittent pain and stiffness develop with use
3. joint inflammation, swelling, reduced mobility increase
4. bone spur formation, deformity, chronic pain, disability may occur
Common Locations: knees, hips, hands, spine, neck, lower back, feet, thumb base, finger joints
Duration: chronic lifelong condition, progresses slowly over years
Severity: mild_to_severe
Complications: chronic pain, reduced mobility, falls risk, muscle wasting, depression, sleep disturbance, joint deformity, loss of independence, disability
Home Remedy: regular low-impact exercise, weight reduction, heat or cold therapy, joint protection, assistive devices, stretching, rest during flare-ups
Avoid: repetitive high-impact joint stress, prolonged inactivity, excessive weight gain, poor posture, unsafe overexertion
When to See a Doctor: persistent joint pain, swelling, reduced movement, difficulty with daily tasks, worsening symptoms, uncertain diagnosis
Emergency Signs: sudden inability to bear weight, hot red swollen joint with fever, severe trauma, sudden weakness or numbness, loss of bladder or bowel control
Prevention: maintain healthy weight, regular exercise, strengthen muscles, prevent joint injuries, ergonomic movement habits, manage repetitive strain
Contagious Period: none
Special Notes: no cure currently exists, symptoms often improve with lifestyle measures and medicines, joint replacement surgery may help severe cases, differs from autoimmune arthritis because it is primarily degenerative
===
Disease: Rheumatoid Arthritis
Aliases: RA, chronic inflammatory arthritis, autoimmune rheumatoid disease
Description: chronic autoimmune inflammatory disorder that primarily attacks joint linings, causing pain, swelling, stiffness, progressive joint damage, deformity, and possible systemic organ involvement
Cause: autoimmune reaction where the immune system attacks synovium, influenced by genetics, hormones, smoking, environmental triggers
Transmission: none
Risk Groups: women, middle-aged adults, smokers, family history, genetically predisposed individuals, people with autoimmune susceptibility
Incubation Period: none
Symptoms:
- symmetrical joint pain, swelling, warmth, tenderness, especially hands, wrists, feet
- prolonged morning stiffness, difficulty moving joints, reduced grip strength
- redness around joints, decreased range of motion
- fatigue, low energy, fever, sweating, poor appetite, weight loss
- rheumatoid nodules under skin near pressure points or joints
- progressive deformity, weakness, impaired walking or hand use in advanced disease
Progression:
1. immune system inflames synovial lining of joints
2. persistent swelling, pain, stiffness, intermittent flares develop
3. cartilage, bone, tendons, ligaments become damaged
4. deformity, disability, extra-articular complications may occur if uncontrolled
Common Locations: hands, wrists, fingers, feet, ankles, knees, elbows, shoulders, neck, other joints
Duration: chronic lifelong condition with relapsing and remitting flares
Severity: moderate_to_severe
Complications: joint destruction, deformity, carpal tunnel syndrome, lung inflammation, heart inflammation, cardiovascular disease, stroke, osteoporosis, depression, disability
Home Remedy: regular gentle exercise, joint protection, balanced rest and activity, heat therapy, stress management, healthy diet, smoking cessation
Avoid: smoking, untreated inflammation, prolonged inactivity, excessive joint strain, stopping prescribed medicines without advice
When to See a Doctor: persistent joint pain or swelling, morning stiffness over several weeks, reduced function, flare symptoms, medication side effects, unexplained fatigue or weight loss
Emergency Signs: chest pain, severe shortness of breath, sudden neurological symptoms, inability to move joint after rapid swelling, high fever with severe flare, loss of bladder or bowel control
Prevention: no guaranteed prevention, avoid smoking, maintain healthy weight, early diagnosis, prompt treatment, regular monitoring
Contagious Period: none
Special Notes: early treatment with DMARDs or biologics can slow damage significantly, symptoms often fluctuate with flares and remissions, pregnancy planning is important because some medicines are unsafe in pregnancy
===
Disease: Low Back Pain
Aliases: back pain, lumbar pain, lumbago, backache, lumbar strain
Description: common symptom involving pain or discomfort in the lower back, usually mechanical and self-limiting, but occasionally caused by nerve compression, injury, infection, cancer, or serious spinal disorders
Cause: muscle strain, ligament sprain, disc degeneration, herniated disc, poor movement patterns, prolonged sitting, awkward lifting, overuse, arthritis, spinal stenosis, trauma, less commonly infection or tumor
Transmission: none
Risk Groups: adults of all ages, sedentary individuals, heavy manual workers, obesity, poor conditioning, older adults, prior back problems, repetitive lifting occupations
Incubation Period: none
Symptoms:
- aching, stiffness, soreness, muscle spasm in lower back
- sharp, burning, stabbing, shooting pain in back or buttock
- pain radiating into one or both legs, sciatica
- pins and needles, numbness, weakness from nerve irritation
- pain worse with bending, lifting, prolonged sitting, certain movements
- reduced mobility, difficulty standing upright, trouble walking or getting out of bed
- severe constant night pain, unexplained weight loss, fever, bladder or bowel changes may indicate serious cause
Progression:
1. sudden or gradual onset of lumbar pain occurs
2. stiffness and movement limitation develop during acute phase
3. most mechanical cases improve within days to weeks with activity and self-care
4. some cases become chronic, recurrent, or reveal underlying structural disease
Common Locations: lower back, lumbar spine, buttocks, hips, back of thigh, legs, feet when nerve-related
Duration: acute days to 6 weeks, subacute 6-12 weeks, chronic over 12 weeks
Severity: mild_to_severe
Complications: chronic pain, reduced mobility, work disability, depression, sleep disturbance, falls risk, persistent sciatica, cauda equina syndrome, neurological deficits
Home Remedy: stay active, gentle walking, gradual exercises, heat or ice packs, posture changes, short-term pain relief medicines if appropriate, avoid prolonged bed rest
Avoid: prolonged inactivity, excessive bed rest, awkward heavy lifting, repeated painful twisting, fear-based immobilization, unnecessary early imaging without warning signs
When to See a Doctor: pain lasting over 6 weeks, worsening pain, leg weakness, numbness, night pain, pain not affected by movement, history of cancer, fever, significant trauma
Emergency Signs: loss of bladder or bowel control, urinary retention, saddle numbness, rapidly worsening leg weakness, inability to walk, severe trauma, fever with severe back pain
Prevention: regular exercise, core and back strengthening, healthy weight, ergonomic lifting, frequent movement breaks, maintain flexibility, stop smoking
Contagious Period: none
Special Notes: most cases do not need early X-ray or MRI, imaging is usually reserved for red flags or persistent symptoms, MRI may show age-related changes that are not the true pain source
===
Disease: Osteomyelitis
Aliases: bone infection
Description: infection and inflammation of bone tissue caused by bacteria, fungi, or other microorganisms, may be acute or chronic and can damage bone and surrounding tissues
Cause: most commonly bacteria, also fungi or other germs, spread from nearby infected tissue, bloodstream, trauma, surgery, implants, or open wounds
Transmission: not usually person-to-person, develops from internal spread of infection, contaminated wounds, surgery, or direct inoculation after injury
Risk Groups: diabetes, hemodialysis patients, poor blood supply, recent injury, injected illicit drug use, bone surgery, prosthetic joints, weakened immune system
Incubation Period: variable, days to weeks after infection source or injury
Symptoms:
- bone pain, pain at site of infection, tenderness
- fever, chills, excessive sweating, malaise, fatigue
- local swelling, redness, warmth over affected area
- open wound with pus or drainage
- chronic cases may have recurring pain and intermittent symptoms
Progression:
1. microorganisms enter bone through blood, nearby tissue, or direct contamination
2. inflammation develops causing pain, swelling, and fever
3. bone tissue damage and reduced blood supply may occur
4. chronic infection, abscess, dead bone, or recurrence may develop if untreated
Common Locations: long bones of arms and legs in children, feet, vertebrae, pelvis, hips in adults, surgical implant sites
Duration: acute cases weeks to months, chronic cases may persist or recur for years
Severity: moderate_to_severe
Complications: chronic osteomyelitis, bone necrosis, abscess, sepsis, fracture, impaired mobility, prosthetic joint failure, amputation
Home Remedy: rest, hydration, wound care, strict adherence to antibiotics, glucose control in diabetes, avoid pressure on affected area
Avoid: delaying treatment, stopping antibiotics early, smoking, uncontrolled diabetes, weight-bearing on affected bone without advice, injecting illicit drugs
When to See a Doctor: persistent bone pain, fever, swelling, draining wound, symptoms after surgery or injury, symptoms not improving with treatment
Emergency Signs: high fever, sepsis signs, severe pain, rapidly spreading redness, inability to move limb, confusion, severe weakness
Prevention: prompt treatment of skin and wound infections, sterile surgical technique, good diabetes control, proper foot care, avoid needle sharing, early care after injuries
Contagious Period: none
Special Notes: diagnosis often requires imaging and blood tests, bone biopsy may identify organism, antibiotics usually needed for 4-6 weeks or longer, surgery may be required to remove dead bone or infected hardware
===
Disease: Fibromyalgia
Aliases: fibromyalgia syndrome, FMS, chronic widespread pain syndrome
Description: chronic pain syndrome characterized by widespread musculoskeletal pain, heightened pain sensitivity, fatigue, sleep disturbance, and cognitive symptoms without ongoing tissue damage
Cause: exact cause unknown, abnormal central pain processing, nervous system sensitization, genetic predisposition, stress-related and environmental triggers
Transmission: none
Risk Groups: women, adults aged 35-60, family history, people with anxiety or depression, chronic stress exposure, obesity, rheumatoid arthritis, lupus, trauma history
Incubation Period: none
Symptoms:
- widespread body pain described as aching, burning, stabbing, or mixed pain
- increased sensitivity to touch, light, temperature, noise, odors, or medicines
- tender points in neck, shoulders, elbows, hips, knees, back of head
- fatigue despite adequate sleep, low energy, poor stamina
- sleep disturbance, unrefreshing sleep, insomnia
- muscle stiffness, spasms, pain that moves around body
- numbness, tingling, pins and needles in hands or feet
- headaches, migraines, jaw pain
- memory or concentration problems, fibro fog
- dizziness, balance issues, irritable bowel syndrome, restless legs syndrome
- anxiety, depression, painful menstrual periods
Progression:
1. widespread pain and fatigue gradually or suddenly begin
2. symptoms fluctuate with stress, weather, activity, sleep quality, illness
3. recurrent flares and remissions occur over time
4. chronic symptoms may persist but can improve with management
Common Locations: widespread muscles and soft tissues, neck, shoulders, back, hips, knees, elbows, head, jaw
Duration: chronic long-term condition, often months to years
Severity: mild_to_severe
Complications: reduced quality of life, sleep disorders, depression, anxiety, reduced work capacity, deconditioning, chronic fatigue, cognitive impairment
Home Remedy: regular gentle exercise, pacing activities, sleep hygiene, stress reduction, relaxation techniques, heat therapy, balanced routine, self-management strategies
Avoid: overexertion, prolonged inactivity, unmanaged stress, irregular sleep patterns, expecting complete bed rest to help, abrupt activity spikes
When to See a Doctor: widespread pain over 3 months, persistent fatigue, poor sleep, memory issues, symptoms affecting daily life, uncertain diagnosis
Emergency Signs: severe suicidal thoughts, chest pain, sudden neurological deficits, high fever, rapidly worsening unexplained symptoms suggesting another condition
Prevention: no proven prevention, regular exercise, stress management, healthy sleep habits, healthy weight, early treatment of associated mood disorders
Contagious Period: none
Special Notes: fibromyalgia does not damage muscles or organs and is not life-threatening, diagnosis is clinical after excluding other causes, pain medicines alone are often less effective than combined lifestyle and multidisciplinary treatment
===
Disease: Scoliosis
Aliases: spinal curvature, lateral spinal curvature, spinal deformity
Description: abnormal sideways curvature and rotation of the spine that may develop in childhood or adulthood, ranging from mild cosmetic change to severe deformity affecting pain, posture, and organ function
Cause: idiopathic most common, genetic predisposition, congenital vertebral malformation, neuromuscular disorders, connective tissue disorders, spinal tumors, age-related degenerative spinal changes
Transmission: none
Risk Groups: children aged 10-15, adolescents during growth spurts, family history, females with adolescent idiopathic scoliosis, people with cerebral palsy, muscular dystrophy, Marfan syndrome, osteogenesis imperfecta, older adults with degenerative spine disease
Incubation Period: none
Symptoms:
- visibly curved spine
- one shoulder higher than the other
- one hip more prominent than the other
- uneven waistline or rib prominence
- clothes hanging unevenly
- back pain, more common in adults
- reduced flexibility or posture imbalance
- leg length difference appearance
- severe cases may cause shortness of breath or fatigue
- numbness, weakness, leg pain if nerves are compressed
Progression:
1. mild spinal curve develops or is first noticed
2. curve may worsen during growth spurts or with degeneration
3. trunk asymmetry, pain, stiffness, functional issues may increase
4. severe untreated curves may impair lungs, heart, or nerves
Common Locations: thoracic spine, lumbar spine, thoracolumbar spine, rib cage, shoulders, hips
Duration: chronic condition, progression varies with age and cause
Severity: mild_to_severe
Complications: chronic back pain, arthritis, body image distress, reduced quality of life, lung restriction, heart strain, pneumonia risk, nerve compression, weakness, disability
Home Remedy: posture awareness, prescribed exercises, maintain fitness, core strengthening, pain management, follow specialist monitoring schedule
Avoid: ignoring progressive curves in children, heavy strain without guidance, poor brace compliance, delaying specialist review for worsening symptoms
When to See a Doctor: visible spinal curve, uneven shoulders or hips, persistent back pain, worsening posture, growth-age child with asymmetry, numbness or weakness
Emergency Signs: new bladder or bowel dysfunction, saddle numbness, rapidly worsening leg weakness, severe trauma, sudden severe neurological symptoms, breathing difficulty with severe deformity
Prevention: many cases cannot be prevented, early screening and monitoring, treat underlying neuromuscular disorders, maintain bone and muscle health
Contagious Period: none
Special Notes: most cases are idiopathic and not caused by poor posture, treatment may include observation, bracing, or surgery depending on curve size and growth remaining, X-ray commonly confirms diagnosis
===
Disease: Tendinitis
Aliases: tendonitis, tendinopathy, tendon inflammation
Description: painful inflammatory or degenerative condition affecting tendons around joints, commonly caused by overuse, injury, or repetitive motion
Cause: repetitive strain, sudden injury, overuse, poor biomechanics, sports activity, occupational stress, aging changes
Transmission: none
Risk Groups: athletes, manual workers, repetitive motion workers, older adults, people with poor posture, prior joint injury
Incubation Period: none
Symptoms:
- localized pain near a joint or tendon
- tenderness to touch over tendon
- pain worsened by movement or resistance
- swelling, stiffness, reduced range of motion
- weakness or difficulty using affected limb
- sudden severe pain or popping may suggest tendon rupture
Progression:
1. overuse or injury irritates tendon
2. inflammation and pain develop with activity
3. persistent irritation causes weakness, stiffness, recurring pain
4. chronic degeneration or rupture may occur if untreated
Common Locations: shoulder, wrist, elbow, neck, hip, knee, ankle, heel, Achilles tendon
Duration: acute days to weeks, chronic cases may last months
Severity: mild_to_moderate
Complications: chronic pain, reduced mobility, tendon rupture, recurrent inflammation, joint stiffness, weakness, work limitation
Home Remedy: rest, ice, compression, elevation, temporary activity reduction, gentle stretching after acute pain improves, ergonomic changes
Avoid: repetitive painful motions, returning to activity too quickly, ignoring worsening symptoms
When to See a Doctor: pain lasting more than a few weeks, recurrent swelling, reduced movement, weakness, suspected infection, uncertain diagnosis
Emergency Signs: fever with red hot swollen joint, inability to move limb, sudden tendon snap, severe swelling, rapidly worsening pain
Prevention: warm up before exercise, stretch regularly, use proper technique, take rest breaks, vary repetitive tasks, strengthen supporting muscles, maintain healthy weight
Contagious Period: none
Special Notes: imaging such as X-ray, ultrasound, or MRI is often unnecessary initially unless symptoms persist or recur, corticosteroid injections may provide short-term relief, surgery is uncommon except ruptures or resistant cases
===
Disease: Bursitis
Aliases: inflamed bursa, bursal inflammation, bursal syndrome
Description: painful inflammation of a bursa, a small fluid-filled sac that cushions bones, tendons, muscles, or skin near joints, commonly caused by overuse or pressure
Cause: repetitive motion, prolonged pressure on joints, kneeling, leaning on elbows, acute injury, friction, infection, gout, inflammatory arthritis
Transmission: none
Risk Groups: athletes, manual workers, people who kneel often, people leaning on elbows, older adults, people with gout, rheumatoid arthritis, diabetes, prior joint injury
Incubation Period: none
Symptoms:
- localized joint pain
- swelling over affected bursa
- tenderness to touch
- pain worsened by movement or pressure
- warmth or redness over area
- stiffness or reduced movement
- pain while resting on affected side or surface
- fever or marked redness may suggest infection
Progression:
1. repeated stress or injury irritates the bursa
2. fluid accumulation and inflammation develop
3. pain and swelling limit joint use
4. chronic thickening, recurrence, or infection may occur if untreated
Common Locations: knee, elbow, shoulder, hip, heel, ankle
Duration: acute days to weeks, chronic or recurrent cases may last months
Severity: mild_to_moderate
Complications: chronic pain, reduced mobility, recurrent bursitis, infection, surrounding tendon irritation, functional limitation
Home Remedy: rest, ice, avoid pressure on joint, compression if appropriate, elevation, gradual return to activity, padding of affected area
Avoid: kneeling on hard surfaces, leaning on elbows, repetitive painful movements, continuing aggravating activity
When to See a Doctor: persistent swelling, severe pain, recurrent episodes, reduced movement, suspected infection, symptoms not improving with rest
Emergency Signs: fever, rapidly increasing redness, severe swelling, inability to move joint, severe pain, spreading infection signs
Prevention: use knee or elbow pads, take movement breaks, proper technique during activity, avoid prolonged pressure, strengthen supporting muscles, manage gout or arthritis
Contagious Period: none
Special Notes: diagnosis is often clinical, fluid aspiration may be used to rule out infection or crystals, steroid injection or surgery may be considered for persistent cases
===
Disease: Carpal Tunnel Syndrome
Aliases: CTS, median nerve compression
Description: common nerve compression disorder caused by pressure on the median nerve within the wrist carpal tunnel, leading to pain, numbness, tingling, weakness, and reduced hand function
Cause: compression of the median nerve in the carpal tunnel due to swelling of tendons, inflammation, repetitive strain, wrist anatomy changes, pregnancy, hormonal changes, or associated medical conditions
Transmission: none
Risk Groups: women, middle-aged or older adults, pregnancy, overweight individuals, repetitive hand workers, diabetes, rheumatoid arthritis, hypothyroidism, menopause, prior wrist fracture
Incubation Period: none
Symptoms:
- numbness, tingling, burning in thumb, index, middle, and part of ring finger
- wrist or hand pain, often worse at night or early morning
- weakness, clumsiness, dropping objects, reduced grip strength
- difficulty with buttoning clothes, holding phone, book, or steering wheel
- shaking hand for relief, sensation of swelling without visible swelling
- severe cases may show thumb muscle wasting and persistent numbness
Progression:
1. intermittent numbness or tingling develops, especially at night
2. symptoms increase with gripping, bending wrist, or repetitive hand use
3. persistent pain, sensory loss, and hand weakness occur
4. untreated severe cases may cause muscle wasting and functional impairment
Common Locations: wrist carpal tunnel, palm side of hand, thumb, index finger, middle finger, radial half of ring finger
Duration: variable, weeks to years depending on cause and treatment
Severity: mild_to_moderate
Complications: chronic pain, persistent numbness, grip weakness, thenar muscle wasting, reduced dexterity, permanent nerve damage if prolonged
Home Remedy: nighttime wrist splinting, activity modification, ergonomic changes, rest breaks, wrist positioning neutral, gentle hand exercises
Avoid: prolonged wrist bending, repetitive squeezing or gripping, delaying treatment when weakness develops, constant splint use without guidance
When to See a Doctor: symptoms lasting more than several weeks, worsening numbness, weakness, dropping objects, sleep disruption, reduced hand function, uncertain diagnosis
Emergency Signs: sudden severe hand weakness, rapidly worsening numbness, loss of hand function, signs of trauma with deformity, severe swelling or infection
Prevention: ergonomic workplace setup, regular breaks from repetitive tasks, neutral wrist posture, weight management, diabetes control, treatment of thyroid or inflammatory disease
Contagious Period: none
Special Notes: diagnosis is mainly clinical and may use Phalen test, Tinel sign, nerve conduction studies, ultrasound, or MRI, pregnancy-related cases often improve after delivery, surgery may be needed when conservative treatment fails
===
Disease: Dermatitis Herpetiformis
Aliases: DH
Description: autoimmune skin condition strongly linked to coeliac disease that causes an intensely itchy and blistering rash
Cause: immune system reaction to the gluten protein found in wheat, barley, and rye
Transmission: none
Risk Groups: people aged 15 to 40, men, individuals with coeliac disease
Incubation Period: none
Symptoms:
- red raised patches on the skin
- blisters that burst with scratching
- severe itching and stinging
- gut symptoms like diarrhoea, constipation, stomach pain, or bloating
Progression:
1. gluten consumption triggers an autoimmune response
2. intensely itchy red patches and blisters develop symmetrically on the body
3. scratching causes blisters to burst and crust
4. strict gluten-free diet gradually heals the skin over up to 2 years
Common Locations: elbows, knees, buttocks, usually symmetrical on both sides of the body
Duration: lifelong condition requiring continuous dietary management, skin healing takes up to 2 years
Severity: moderate
Complications: osteoporosis, gut cancer, increased risk of autoimmune diseases like type 1 diabetes and thyroid disease
Home Remedy: strict lifelong gluten-free diet
Avoid: foods containing gluten such as wheat, barley, rye, and sometimes oats
When to See a Doctor: red blistering patches appear, suspected coeliac disease
Emergency Signs: none
Prevention: strictly follow a gluten-free diet to prevent rash flare-ups and complications
Contagious Period: none
Special Notes: diagnosis requires a skin biopsy while the patient is still consuming gluten, medication like Dapsone can control itching but has side effects like anaemia
===
Disease: Rashes
Aliases: dermatitis, skin rash, contact dermatitis
Description: an area of irritated or swollen skin that can be itchy, red, painful, and may lead to blisters or patches of raw skin
Cause: irritating substances, allergies, chemicals, poison ivy, infections, medications, or genetic predisposition
Transmission: none
Risk Groups: people with specific allergies or genetic predisposition
Incubation Period: can develop right away or over several days depending on the trigger
Symptoms:
- itchy, red, painful, or irritated skin
- small bumps
- blisters or patches of raw skin
Progression:
1. skin is exposed to an irritant, allergen, or underlying trigger
2. redness, itching, and swelling develop immediately or over several days
3. bumps, blisters, or raw patches may form
Common Locations: anywhere on the body exposed to irritants or allergens
Duration: most clear up quickly, though some require long-term treatment
Severity: mild_to_moderate
Complications: secondary infection, scarring, chronic skin changes
Home Remedy: moisturizers, lotions, baths, cortisone creams, antihistamines
Avoid: contact with known irritants or allergens
When to See a Doctor: rash is severe, does not go away, or is accompanied by other symptoms
Emergency Signs: none
Prevention: avoid touching chemicals, poison ivy, or known allergens
Contagious Period: none
Special Notes: identifying the specific cause of the rash is necessary before starting treatment
===
Disease: Atopic Eczema
Aliases: atopic dermatitis, eczema
Description: chronic inflammatory skin condition characterized by itchy, dry, and inflamed skin associated with immune dysregulation and barrier dysfunction
Cause: genetic predisposition, immune system overactivity, skin barrier defects, environmental triggers, allergens
Transmission: none
Risk Groups: children, individuals with family history of eczema, asthma or allergies, people with sensitive skin
Incubation Period: none
Symptoms:
- intense itching (pruritus)
- dry, cracked skin
- red or inflamed patches
- thickened or scaly skin in chronic cases
- oozing or crusting in severe cases
Progression:
1. skin barrier dysfunction and dryness
2. exposure to triggers leading to inflammation
3. itching and scratching cycle
4. chronic skin changes and flare-ups
Common Locations: face, neck, elbows, knees, hands, wrists
Duration: chronic with periodic flare-ups
Severity: mild_to_moderate
Complications: skin infections, sleep disturbance, scarring, psychological distress
Home Remedy: moisturizers, avoiding triggers, gentle skincare, cool compress
Avoid: scratching, harsh soaps, allergens, extreme temperatures
When to See a Doctor: persistent or severe symptoms, infection signs, poor response to treatment
Emergency Signs: widespread infection, severe swelling, high fever
Prevention: regular moisturizing, trigger avoidance, maintaining skin barrier, proper hygiene
Contagious Period: none
Special Notes: often associated with asthma and allergic rhinitis (atopic triad), early management reduces severity, relapsing condition
===
Disease: Melasma
Aliases: chloasma, mask of pregnancy, pregnancy mask
Description: a common pigmentation disorder characterized by symmetrical brown or gray-brown patches on sun-exposed areas of the face, often associated with hormonal changes and ultraviolet (UV) exposure
Cause: increased melanin production triggered by hormonal changes (estrogen and progesterone), sun exposure (UV radiation), genetic predisposition, use of hormonal medications (e.g., oral contraceptives, hormone replacement therapy)
Transmission: none
Risk Groups: women (especially of reproductive age), pregnant women, individuals with darker skin tones, people living in tropical or high sun-exposure regions, users of hormonal therapies or birth control pills
Incubation Period: none
Symptoms:
- symmetrical dark patches on face (cheeks, forehead, nose, upper lip)
- brown, gray-brown, or tan discoloration
- no itching or pain
- cosmetic concern or emotional distress
- gradual worsening with sun exposure
Progression:
1. trigger exposure (hormones or sunlight)
2. increased melanin production in skin
3. formation of dark patches on face
4. may fade or recur depending on triggers
Common Locations: cheeks, forehead, bridge of nose, upper lip, chin
Duration: months to years (may fade after pregnancy or stopping hormones)
Severity: mild_to_moderate
Complications: psychological distress or reduced self-esteem
Home Remedy: daily use of sunscreen (SPF 30 or higher), wearing hats and protective clothing, avoiding direct sun exposure
Avoid: excessive sunlight or UV exposure, tanning beds and sun lamps, unnecessary hormonal medications (if medically safe to stop)
When to See a Doctor: persistent or worsening facial pigmentation, uncertainty about diagnosis, significant cosmetic or emotional concern
Emergency Signs: none
Prevention: consistent sun protection (broad-spectrum sunscreen), protective clothing and shade, avoiding peak sunlight hours, careful use of hormonal therapies
Contagious Period: none
Special Notes: often improves after pregnancy or discontinuation of hormonal triggers, recurrence is common with sun exposure, treatments include topical creams, chemical peels, and laser therapy, diagnosis may involve examination with a Wood's lamp to assess pigmentation depth
===
"""

# ─────────────────────────────────────────────────────────────────────────────
# PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_diseases(text: str) -> list[dict]:
    blocks = [b.strip() for b in text.split("===") if b.strip()]
    diseases = []
    for block in blocks:
        lines = block.splitlines()
        d = {}
        current_key = None
        current_list = []
        list_keys = {"Symptoms", "Progression", "Complications",
                     "Home Remedy", "Avoid", "Prevention"}

        def flush():
            if current_key and current_list:
                d[current_key] = current_list if current_key in list_keys else " ".join(current_list)

        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Check for key: value pattern
            m = re.match(r'^([A-Za-z][A-Za-z\s/()]+?):\s*(.*)', line)
            if m:
                flush()
                current_list = []
                current_key = m.group(1).strip()
                val = m.group(2).strip()
                if val:
                    current_list = [val]
            elif line.startswith("-") or re.match(r'^\d+\.', line):
                item = re.sub(r'^[-\d.]+\s*', '', line).strip()
                if item:
                    current_list.append(item)
            else:
                if current_list:
                    current_list[-1] += " " + line
                else:
                    current_list = [line]
        flush()

        if "Disease" in d:
            diseases.append(d)
    return diseases


# ─────────────────────────────────────────────────────────────────────────────
# BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def build_disease_templates(diseases):
    templates = {}
    for d in diseases:
        name = d.get("Disease", "Unknown")
        templates[name] = {
            "aliases": [a.strip() for a in d.get("Aliases", "").split(",") if a.strip()],
            "description": d.get("Description", ""),
            "cause": d.get("Cause", ""),
            "transmission": d.get("Transmission", "none"),
            "risk_groups": d.get("Risk Groups", ""),
            "incubation_period": d.get("Incubation Period", "none"),
            "symptoms": d.get("Symptoms", []) if isinstance(d.get("Symptoms"), list) else [],
            "progression": d.get("Progression", []) if isinstance(d.get("Progression"), list) else [],
            "common_locations": d.get("Common Locations", ""),
            "duration": d.get("Duration", ""),
            "severity": d.get("Severity", "unknown"),
            "complications": d.get("Complications", []) if isinstance(d.get("Complications"), list) else [],
            "home_remedy": d.get("Home Remedy", []) if isinstance(d.get("Home Remedy"), list) else [],
            "avoid": d.get("Avoid", []) if isinstance(d.get("Avoid"), list) else [],
            "when_to_see_doctor": d.get("When to See a Doctor", ""),
            "emergency_signs": d.get("Emergency Signs", ""),
            "prevention": d.get("Prevention", []) if isinstance(d.get("Prevention"), list) else [],
            "contagious_period": d.get("Contagious Period", "none"),
            "special_notes": d.get("Special Notes", ""),
        }
    return templates


def build_symptom_vocab(diseases):
    vocab = {}
    idx = 0
    for d in diseases:
        symptoms = d.get("Symptoms", [])
        if isinstance(symptoms, list):
            for s in symptoms:
                tokens = re.findall(r'\b\w+\b', s.lower())
                for t in tokens:
                    if t not in vocab and len(t) > 2:
                        vocab[t] = idx
                        idx += 1
    return vocab


def build_label_encoder(diseases):
    names = sorted(set(d.get("Disease", "Unknown") for d in diseases))
    return {"classes": names, "label_to_id": {n: i for i, n in enumerate(names)},
            "id_to_label": {str(i): n for i, n in enumerate(names)}}


def build_feature_dictionary(diseases):
    feat = defaultdict(set)
    for d in diseases:
        name = d.get("Disease", "Unknown")
        symptoms = d.get("Symptoms", [])
        if isinstance(symptoms, list):
            for s in symptoms:
                tokens = re.findall(r'\b\w+\b', s.lower())
                for t in tokens:
                    if len(t) > 2:
                        feat[t].add(name)
    return {k: sorted(v) for k, v in feat.items()}


def build_class_distribution(diseases):
    dist = {}
    total = len(diseases)
    for d in diseases:
        name = d.get("Disease", "Unknown")
        sev = d.get("Severity", "unknown")
        dist[name] = {"severity": sev, "relative_frequency": round(1 / total, 6)}
    return dist


def build_rag_db(diseases):
    db = []
    for d in diseases:
        name = d.get("Disease", "Unknown")
        symptoms = d.get("Symptoms", [])
        complications = d.get("Complications", [])
        chunk = {
            "disease": name,
            "aliases": [a.strip() for a in d.get("Aliases", "").split(",") if a.strip()],
            "description": d.get("Description", ""),
            "symptoms_text": "; ".join(symptoms) if isinstance(symptoms, list) else str(symptoms),
            "cause": d.get("Cause", ""),
            "severity": d.get("Severity", "unknown"),
            "transmission": d.get("Transmission", "none"),
            "emergency_signs": d.get("Emergency Signs", ""),
            "complications_text": "; ".join(complications) if isinstance(complications, list) else str(complications),
            "prevention": "; ".join(d.get("Prevention", [])) if isinstance(d.get("Prevention"), list) else str(d.get("Prevention", "")),
        }
        db.append(chunk)
    return db


def generate_csv_rows(diseases, symptom_vocab):
    rows = []
    label_enc = build_label_encoder(diseases)
    for d in diseases:
        name = d.get("Disease", "Unknown")
        label = label_enc["label_to_id"].get(name, -1)
        symptoms = d.get("Symptoms", [])
        vec = [0] * min(len(symptom_vocab), 200)
        if isinstance(symptoms, list):
            for s in symptoms:
                tokens = re.findall(r'\b\w+\b', s.lower())
                for t in tokens:
                    if t in symptom_vocab and symptom_vocab[t] < 200:
                        vec[symptom_vocab[t]] = 1
        rows.append({"disease": name, "label": label,
                     "severity": d.get("Severity", "unknown"),
                     "symptom_vector": ",".join(map(str, vec)),
                     "symptom_count": sum(vec)})
    return rows


def write_rag_chunks(diseases, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for d in diseases:
        name = d.get("Disease", "Unknown")
        safe = re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_').lower()
        symptoms = d.get("Symptoms", [])
        complications = d.get("Complications", [])
        home = d.get("Home Remedy", [])
        avoid = d.get("Avoid", [])
        prevention = d.get("Prevention", [])

        def fmt(v):
            if isinstance(v, list):
                return "\n".join(f"- {i}" for i in v)
            return str(v)

        content = f"""# {name}

**Aliases:** {d.get('Aliases', 'none')}

## Description
{d.get('Description', '')}

## Cause
{d.get('Cause', '')}

## Transmission
{d.get('Transmission', 'none')}

## Risk Groups
{d.get('Risk Groups', '')}

## Incubation Period
{d.get('Incubation Period', 'none')}

## Symptoms
{fmt(symptoms)}

## Progression
{fmt(d.get('Progression', []))}

## Common Locations
{d.get('Common Locations', '')}

## Duration
{d.get('Duration', '')}

## Severity
{d.get('Severity', 'unknown')}

## Complications
{fmt(complications)}

## Home Remedies
{fmt(home)}

## Things to Avoid
{fmt(avoid)}

## When to See a Doctor
{d.get('When to See a Doctor', '')}

## Emergency Signs
{d.get('Emergency Signs', '')}

## Prevention
{fmt(prevention)}

## Contagious Period
{d.get('Contagious Period', 'none')}

## Special Notes
{d.get('Special Notes', '')}
"""
        with open(os.path.join(out_dir, f"{safe}.md"), "w", encoding="utf-8") as f:
            f.write(content)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Combine both parts of the disease text
    # PART 1 is in DISEASE_TEXT (defined earlier in the original script)
    # PART 2 is defined above — merge them before parsing
    full_text = DISEASE_TEXT + "\n" + DISEASE_TEXT_PART2

    diseases = parse_diseases(full_text)
    diseases = [d for d in diseases if not is_excluded_disease(d.get("Disease", ""))]
    print(f"Parsed {len(diseases)} diseases.")

    out = "output"
    os.makedirs(out, exist_ok=True)

    templates = build_disease_templates(diseases)
    with open(f"{out}/disease_templates.json", "w", encoding="utf-8") as f:
        json.dump(templates, f, indent=2, ensure_ascii=False)

    rag_db = build_rag_db(diseases)
    with open(f"{out}/rag_disease_db.json", "w", encoding="utf-8") as f:
        json.dump(rag_db, f, indent=2, ensure_ascii=False)

    vocab = build_symptom_vocab(diseases)
    with open(f"{out}/symptom_vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)

    enc = build_label_encoder(diseases)
    with open(f"{out}/label_encoder.json", "w", encoding="utf-8") as f:
        json.dump(enc, f, indent=2, ensure_ascii=False)

    feat_dict = build_feature_dictionary(diseases)
    with open(f"{out}/feature_dictionary.json", "w", encoding="utf-8") as f:
        json.dump(feat_dict, f, indent=2, ensure_ascii=False)

    cls_dist = build_class_distribution(diseases)
    with open(f"{out}/class_distribution.json", "w", encoding="utf-8") as f:
        json.dump(cls_dist, f, indent=2, ensure_ascii=False)

    rows = generate_csv_rows(diseases, vocab)
    random.shuffle(rows)
    split = int(len(rows) * 0.8)
    train_rows, test_rows = rows[:split], rows[split:]

    fieldnames = ["disease", "label", "severity", "symptom_vector", "symptom_count"]
    for fname, data in [("train.csv", train_rows), ("test.csv", test_rows)]:
        with open(f"{out}/{fname}", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

    write_rag_chunks(diseases, f"{out}/rag_chunks")

    print(f"Done! Files written to '{out}/':")
    print(f"  disease_templates.json  — {len(templates)} diseases")
    print(f"  rag_disease_db.json     — {len(rag_db)} entries")
    print(f"  symptom_vocab.json      — {len(vocab)} tokens")
    print(f"  label_encoder.json      — {len(enc['classes'])} classes")
    print(f"  feature_dictionary.json — {len(feat_dict)} features")
    print(f"  class_distribution.json — {len(cls_dist)} entries")
    print(f"  train.csv               — {len(train_rows)} rows")
    print(f"  test.csv                — {len(test_rows)} rows")
    print(f"  rag_chunks/             — {len(diseases)} .md files")


if __name__ == "__main__":
    main()
