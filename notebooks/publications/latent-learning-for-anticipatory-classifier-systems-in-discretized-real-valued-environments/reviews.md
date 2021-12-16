# Reviewers remarks

----------------------

## Reviewer 1

### Lack of statistical evaluation ✅

**Response to concern**: We agree that the paper lacks the statistical evaluation. We decided to incorporate Bayesian Estimation of all metrics obtained in the agents last trial. This approach provides more information (like the magnitude of difference) than traditional t-test. Additionally all experiments were repeated fixed number of 50 times and then averaged. We have chosen this number, because due to the central limit theorem we can assume the normal distribution of results for later Bayesian Analysis.

**Action to remedy the concern**: Added section related to statistical testing.

### Reproducible repository code 📝

**Response to concern**: Agree. All used repositories were tagged to indicate precise point in time when the calculations were obtained.

**Action to remedy the concern**: Create Git tags in 3 repositories and reference them explicitly in paper.


### Inconsistent latent learning definition ✅

> Further, I can't quite follow the process of how the latent learning capabilities are measured. Latent learning means learning independent of the reward (this is how it is explained in the manuscript's introduction). In the scenarios considered, there is, however, always a reward received. It seems that this study does not measure latent learning (at least not by the definition given in the introduction) but instead the environment model-building capabilities of the methods investigated, which also seems to be what the chosen metrics measure. This should be clearly stated in the manuscript's title as well as when presenting the work's objective in the introduction.

**Response to concern**: That is right. As stated in the paper all methods operate in explore mode, meaning that they take no advantage of the environmental reward. However, the title is misleading though. 

**Action to remedy the concern**: Changed the paper title to indicate the capabilities of building internal representation of the environmental model.

### Incorrect description of agent goal ✅

> “The agent's goal is to create an internal model … of the environment by estimating …” This is not correct. An RL agent's goal is to maximize the return (sum of rewards) and one *means* to do that is to perform the mentioned estimates. There are several other options that are widely applied to RL problems (e.g. q-learning which does not estimate the transition function explicitly or direct policy approximation which does not perform either of the two estimates).

**Response to concern**: Agree, this statement is not correct.

**Action to remedy the concern**: Update the introduction with stating proper agent goal. 


### Parameters settings ✅

> - [x] Isn't roulette wheel selection deprecated in favour of tournament selection in most cases? Why did was roulette wheel selection chosen for ACS2?
> - [x] Why does Dyna-Q perform 5 steps ahead simulation? Why not more? Why not less?
> - [x] Are the scenarios similar enough that the same parameter settings are applicable to all (especially a how many steps to look ahead ideally in Dyna-Q seems to depend strongly on the number of steps episodes typically require)?

**Response to concern**: For the offspring selection the roulette-wheel method is proposed by Butz - to our knowledge no others were investigated in ACS2 algorithms. For the Dyna-Q planning 5 steps ahead seems reasonable for the investigated environments (20-Corridor, 20x20 Grid, Simple Maze) - we expected to see some noticeable improvement here. Regarding other parameter settings - we configured the algorithms to defaults used in most LCS papers. They might be a bit sweepy, but provide a baseline for rough comparison with other research. The proper parameter tuning was out of the scope for this work. 

**Action to remedy the concern**: None taken

### Missing reward schemes ✅

> The authors don't specify the reward schemes for neither rMPX nor simple maze

**Response to concern**: Agree

**Action to remedy the concern**: We added reward schemes for rMPX and Simple Maze in environmental section.

### Vague summary statement ✅

> “in this work we showed the importance of latent learning” I disagree. Isn't a comparison with learners that do not use latent learning required in order to show its importance?

**Response to concern**: Agree

**Action to remedy the concern**: Rephrased the sentence

### Language, spelling, formatting 📝

> - [x] I don't think that anticipatory classifier systems “successfully model conditional rules” but rather that they “successfully build models consisting of conditional rules” or just “successfully build sets of conditional rules”.
> - [x] The general description of LCSs is actually a general description of Michigan-style LCSs. I recommend that this is stated clearly.
> - [x] The meaning of “classifier fitness” (How is that value used in an LCS?) has to be introduced before introducing accuracy-based fitness.
> - [x] Figure 1 has to be explained, it cannot stand for itself. Also, why does the reinforcement arrow end on an arrow between two symbols?
> - [x] The description of YACS is not comprehensible and should be reconsidered/rewritten. How does the DE mechanism work? What is the trace property?
> - [x] It doesn't get entirely clear why the rMPX needs to be extended with an extra bit. Please spend another sentence on that (maybe an example would help).
> - [x] The simple maze in [4] contains walls (as does Fig. 5) but they are not explained (neither how the agent interacts with them) in the text. This definitely needs to be added.
> - [x] In order to have a mostly self-contained article, I recommend to spend a sentence on how OIQ works.
> - [x] Bibliography needs to be formatted properly/consistently
> - [x] Figures: Light yellow on white is barely visible (actually not visible at all for all but the bar diagrams), the colors in the figures require adjustment.
> - [x] Figures: The lines are too thin in general and the symbols used need to be made slightly larger so they can be discerned properly.

**Response to concern**: We agree that there are some inconsistencies and language errors.

**Action to remedy the concern**: Extra proofreading in terms of language and the major following changes:
- defined "classifier fitness",
- explained Figure 1, 
- rewrote parts with YACS description
- changed colormap in plots to be better distinguishable
- change plot styling
- bibliography: fixed lower cased abbreviations
- bibliography: first names abbreviated with 'dot'

-----------------

## Reviewer 2

### Fine-tuning Abstract section ✅

> - The abstract is too limited, not clearly show the whole summary of the paper.
> - Acronyms in the abstract should be fully written, ACS, ACS2, YACS...
 
**Response to concern**: Agree

**Action to remedy the concern**: Abstract was partially rewritten.


### Bibliography items ordering ✅
> In the introduction section, the citation starts with ref [18]. why/How?

**Response to concern**: Agree

**Action to remedy the concern**: Citations were sorted by appearance

### The real data collection environment is not elaborated. It's better to draw the structure. ❓

**Response to concern**: ...

**Action to remedy the concern**: ...

### Unclear description of Figure 2 ✅

> On Figure 2, is it a proposed method? If it is a proposed method, the agent-replay memory-reward relation is not elaborated. No punishment action in this structure?

**Response to concern**: Figure 2 depicts the components interaction in one of the investigated methods. The knowledge is stored as a set of rules in the population of classifiers. There is no punishment in this method.

**Action to remedy the concern**: Highlight the memory-reward relation in figure caption.

### Try to identify, If the action/agent value has not a value of the reward, where is the punishment value will be updated? ✅

**Response to concern**: All of the investigated models support the distribution of rewards over created set of rules. The punishment scheme was not examined, and is out of the scope for this work. Additionally, the obtained reward is not relevant, because the internalized environmental model is learned without and explicit present of the reward (latent learning). 

**Action to remedy the concern**: None taken

### How to classify the corridor and grid points for this work? ✅

**Response to concern**: These are the standard benchmarking problems used in LCS.

**Action to remedy the concern**: None taken

### Unclear figures ✅
> - Figures are not clearly plotted
> - Inconsistent layout/caption of figures.
> - No caption for all figures such as fig.6, 7,8...

**Response to concern**: Agree.

**Action to remedy the concern**: Changed the colormap to be more distinguishable. Lines and markers are also bigger

### Is the study applies multi-agent and action values. What are really agents are represented? ✅

**Response to concern**: This topic is not in the scope of the work.

**Action to remedy the concern**: None taken

### Stale references ✅

> Most of the reference lists are too old. Try to update it.

**Response to concern**: Agree

**Action to remedy the concern**: Updated bibliography with advancements related to latent learning in RL and curiosity motivational mechanism

-----------------

## Reviewer 3

### Language 📝

> The article needs proofreading by a native speaker.

**Response to concern**: Agree

**Action to remedy the concern**: Article was checked and corrected by another person

### Stale references ✅

> The article needs to elaborate with more recent references

**Response to concern**: Agree

**Action to remedy the concern**: Updated bibliography with advancements related to latent learning in RL and curiosity motivational mechanism

-----------------

## Reviewer 4
### Highlight novelty ✅

> Try to elaborate the novelty of this work. Why this work is important in the field of Latent learning.

**Response to concern**: The novelty is highlighted in the first two paragraph in Section V.

**Action to remedy the concern**: None

#TODO
- [ ] annotate pyalcs-experiments repository
- [ ] article proof-reading
- [x] newer bibliography items
- [x] update plots
- [X] check why there is lowercase "xcs" in biography