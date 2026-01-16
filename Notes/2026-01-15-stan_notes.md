shoudl have a full look through to get an idea of the whole project

- play around with the jupyter notebook 


**transmission spectroscopy metric**
how well would we be able to see atmpspheric features
- calculating equilibrium temperature (the temp it would be if emitting as a black body to produce luminosity that we see)
- high TSM - see more stuff

higher TSM - the less the atmos is blcoking or reflecting light - the more light is able to pass through it from the star and therefore reach us

#### first - find out typical parameters of sub neptunes so can filter out the unneccesary of the 7000 planets

- line between sub neptune and super earth is kinda fuzzy
- paper will say 'chose this radius cut and this mass cut'
	- read literature
	- find numbers for typical radii and massess of SN
	- filtering initially by that
	- and then from stellear param calc EQ TEMP 
	- from eq temp calc spectroscopy metric
	- and then give 100 highest ones?
	- but need to select ones with worse spec metric
	- sometimes will have high TSM unexpectedly

could create a list and say 'for example we're spanning these parameter ranges'
but also 'we'd like some lower ones to justify parameter ranges and to find what is impossible

- get them to model wider range?

options for filtering:
- filter directly on the archive
- **download all of the data as a CSV**
- ignore anything that thinks circumbinary planet

note - some planets wont have all parameters needed - so can filter out the ones that we'll never be able to properly use

will need:
- radius
- mass (where available) and if all you have is inclination angle that's fine
	- (sometime cannot get absolute mass and get a constraint from RV)
- eq temp (from stellar parameters - will always be there (almost all with TESS))
	- take all of stellar parameters 
	- orbital period 
	- semimajor axis 
- radius of star
- might want eccentricity
- leave cont flag or sol or stel par ref


##### so what are we doing?
select sample targets initially 
get data 
try fit trends (see if statistically significant for that trend) (probably after 2 weeks or so)
- then try figure out min targets and time to get trend out 

need some filtering based on type planet
- what to chuck out 

reading literature
metric one 
sample selection 
(first 4)
j - band magnitude is something we can get if not in archive 
![[Pasted image 20260115151038.png|300]] 
first most important - review paper and metric
- sample selection paper 


calculate equilbrium temperatures (write a function)

start building tools to analyse data 


management plan with roughly ordered list of sequence of tasks 

peak finding algorithm but scipy doesnt behave great when edge of peak not flat 

#### initial tasks
- deciding on the wide mass/radius limits for just chucking everything out
- reading and summarising: what we will directly use + what the rest of us need to understand from the paper 
	- metric to determine best targets
	- review paper on sub neptune exoplanets
- calculate equilbrium temperatures (write a function)
	- feeds into the TSM stuff x

(surely you need to read the paper to calc the temps? yeah but only that first equation)
- start building tools to analyse data 

calculating equilib


likely a program to observe each once 

need full transit and bits either side (if length is shorter than 6 hours, want same time on each side, (eg 2 hour transit, 1h on each side)) but if longer, only need 3hours max 


#### management plan
- key deliverables (report, slides for seminar, non-assessed worksheet)
	- all stuff getting marks for 
- NOT SUPER fine grained 
	- plan to current extent - don't know unknowns 
- how we are going to pass data between, or at least planning

- milestones eg team 2 nice planet team 1 model team 2 find peaks
	- so get the workflow going for one and can then expand 
- tasks - concrete NOT 'working on code' but 'by __ deadline would like to have code that can take this and do this' - think about baseline that we need 
- remember to plan in the writing part of the report 
- plan in checkpoints - e.g want this workflow to be ready by week ____
	- and then a week before this does this still seem feasible 