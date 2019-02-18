library(tidyverse)
library(pdist)
library(gplots)
library(ggcorrplot)
library(lpSolve)
library(lpSolveAPI)

metabolite_names <- c("3-Phospho-D-glyceroyl phosphate", "D-Glycerate 2-phosphate", "3-Phospho-D-glycerate", "6-Phospho-D-gluconate", "6-phospho-D-glucono-1,5-lactone", "Acetate", "Acetate (ex)", "Acetaldehyde", "Acetaldehyde (ex)", "Acetyl-CoA", "Cis-Aconitate", "Acetyl phosphate", "ADP C10H12N5O10P2", "2-Oxoglutarate", "2-Oxoglutarate (ex)", "AMP C10H12N5O7P", "ATP C10H12N5O13P3", "Citrate", "CO2 CO2", "CO2 CO2 (ex)", "Coenzyme A", "Dihydroxyacetone phosphate", "D-Erythrose 4-phosphate", "Ethanol", "Ethanol (ex)", "D-Fructose 6-phosphate", "D-Fructose 1,6-bisphosphate", "Formate", "Formate (ex)", "D-Fructose", "Fumarate", "Fumarate (ex)", "Glyceraldehyde 3-phosphate", "D-Glucose 6-phosphate", "D-Glucose (ex)", "L-Glutamine", "L-Glutamine (ex)", "L-Glutamate", "L-Glutamate (ex)", "Glyoxylate", "H2O H2O", "H2O H2O (ex)", "H+", "H+ (ex)", "Isocitrate", "D-Lactate", "D-Lactate (ex)", "L-Malate", "L-Malate (ex)", "Nicotinamide adenine dinucleotide", "Nicotinamide adenine dinucleotide - reduced", "Nicotinamide adenine dinucleotide phosphate", "Nicotinamide adenine dinucleotide phosphate - reduced", "Ammonium", "Ammonium (ex)", "O2 O2", "O2 O2 (ex)", "Oxaloacetate", "Phosphoenolpyruvate", "Phosphate", "Phosphate (ex)", "Pyruvate", "Pyruvate (ex)", "Ubiquinone-8", "Ubiquinol-8", "Alpha-D-Ribose 5-phosphate", "D-Ribulose 5-phosphate", "Sedoheptulose 7-phosphate", "Succinate", "Succinate (ex)", "Succinyl-CoA", "D-Xylulose 5-phosphate", "Virtual objective metabolite")
reaction_names <- c("Acetaldehyde dehydrogenase (acetylating)", "Acetaldehyde reversible transport", "Acetate kinase", "Aconitase (half-reaction A, Citrate hydro-lyase)", "Aconitase (half-reaction B, Isocitrate hydro-lyase)", "Acetate reversible transport via proton symport", "Adenylate kinase", "2-Oxogluterate dehydrogenase", "2 oxoglutarate reversible transport via symport", "Alcohol dehydrogenase (ethanol)", "ATP maintenance requirement", "ATP synthase (four protons for one ATP)", "Biomass Objective Function with GAM", "CO2 transporter via diffusion", "Citrate synthase", "Cytochrome oxidase bd (ubiquinol-8: 2 protons)", "D lactate transport via proton symport", "Enolase", "Ethanol reversible transport via proton symport", "Acetate exchange", "Acetaldehyde exchange", "2-Oxoglutarate exchange", "CO2 exchange", "Ethanol exchange", "Formate exchange", "D-Fructose exchange", "Fumarate exchange", "D-Glucose exchange", "L-Glutamine exchange", "L-Glutamate exchange", "H+ exchange", "H2O exchange", "D-lactate exchange", "L-Malate exchange", "Ammonia exchange", "O2 exchange", "Phosphate exchange", "Pyruvate exchange", "Succinate exchange", "Fructose-bisphosphate aldolase", "Fructose-bisphosphatase", "Formate transport in via proton symport", "Formate transport via diffusion", "Fumarate reductase", "Fructose transport via PEP:Pyr PTS (f6p generating)", "Fumarase", "Fumarate transport via proton symport (2 H)", "Glucose 6-phosphate dehydrogenase", "Glyceraldehyde-3-phosphate dehydrogenase", "D-glucose transport via PEP:Pyr PTS", "Glutamine synthetase", "L-glutamine transport via ABC system", "Glutamate dehydrogenase (NADP)", "Glutaminase", "Glutamate synthase (NADPH)", "L glutamate transport via proton symport  reversible", "Phosphogluconate dehydrogenase", "H2O transport via diffusion", "Isocitrate dehydrogenase (NADP)", "Isocitrate lyase", "D-lactate dehydrogenase", "Malate synthase", "Malate transport via proton symport (2 H)", "Malate dehydrogenase", "Malic enzyme (NAD)", "Malic enzyme (NADP)", "NADH dehydrogenase (ubiquinone-8 & 3 protons)", "NAD transhydrogenase", "Ammonia reversible transport", "O2 transport  diffusion ", "Pyruvate dehydrogenase", "Phosphofructokinase", "Pyruvate formate lyase", "Glucose-6-phosphate isomerase", "Phosphoglycerate kinase", "6-phosphogluconolactonase", "Phosphoglycerate mutase", "Phosphate reversible transport via symport", "Phosphoenolpyruvate carboxylase", "Phosphoenolpyruvate carboxykinase", "Phosphoenolpyruvate synthase", "Phosphotransacetylase", "Pyruvate kinase", "Pyruvate transport in via proton symport", "Ribulose 5-phosphate 3-epimerase", "Ribose-5-phosphate isomerase", "Succinate transport via proton symport (2 H)", "Succinate transport out via proton antiport", "Succinate dehydrogenase (irreversible)", "Succinyl-CoA synthetase (ADP-forming)", "Transaldolase", "NAD(P) transhydrogenase", "Transketolase", "Transketolase", "Triose-phosphate isomerase")

# EFMs as calculated by efmtool
efms <- read.csv('/Users/tom/Git/ecmtool/data/e_coli_core_efms.csv', header = FALSE, stringsAsFactors = FALSE)
colnames(efms) <- reaction_names
efms_growth <- efms[efms$`Biomass Objective Function with GAM` > 10^-6,]
efms_normalised <- efms_growth / efms_growth$`Biomass Objective Function with GAM`
efms_compact <- efms_growth[, colSums(efms_growth) != 0]

# Stoichiometry as dumped by ecmtool, FBA as calculated by CBMPy
N <- read.csv('/Users/tom/Git/ecmtool/data/N_all.csv', header = FALSE, stringsAsFactors = FALSE)
fba_fluxes_gluc <- read.csv('/Users/tom/Git/ecmtool/data/fba_fluxes_gluc.csv', header = FALSE, stringsAsFactors = FALSE)
fba_normalised <- fba_fluxes_gluc / fba_fluxes_gluc[,13]
internal_reactions = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95)
colnames(fba_normalised) <- reaction_names

# Determine EFM with lowest euclidean distance to FBA result
fba_squared_dist_normalised <- as.matrix(pdist(efms_normalised, fba_normalised))
closest_efm_index <- which.min(fba_squared_dist_normalised)

# ECMs calculated from EFMs
ecms_calculated <- as.data.frame(t(as.matrix(N[,internal_reactions]) %*% t(efms_normalised[, internal_reactions])))
colnames(ecms_calculated) <- metabolite_names
ecms_calculated_unique <- ecms_calculated %>% round(1) %>% distinct()

# Heatmap of ECMs calculated from efmtool's EFMs
heatmap.2(as.matrix(ecms_calculated[1:100,]), Rowv=FALSE, Colv=FALSE, dendrogram='none', trace='none', key=FALSE,lwid = c(.01,.99),lhei = c(.01,.99),margins = c(5,15 ))

# Determine ECM that belongs to the EFM that corresponds to the FBA result
closest_ecm_calc <- ecms_calculated[closest_efm_index,]
colnames(closest_ecm_calc) <- metabolite_names
closest_ecm_calc_normalised <- closest_ecm_calc / closest_ecm_calc$`Virtual objective metabolite`

# ECMs as calculated by means of the conversion cone in ecmtool
ecms <- read.csv('/Users/tom/Git/ecmtool/conversion_cone.csv', header = FALSE, stringsAsFactors = FALSE)
colnames(ecms) <- metabolite_names
ecms_growth <- ecms[ecms$`Virtual objective metabolite` > 10^-6,]
ecms_normalised <- ecms_growth / ecms_growth$`Virtual objective metabolite`
ecm_squared_dist_normalised <- as.matrix(pdist(ecms_normalised, closest_ecm_calc_normalised))
View(sort(ecm_squared_dist_normalised))
ecms_diff <- sweep(as.matrix(ecms_normalised), 1, as.matrix(closest_ecm_calc_normalised), '-')

# Difference between conversion cone and ECMs calculated from efmtool's EFM matching with FBA result
heatmap.2(ecms_diff, Rowv=FALSE, Colv=FALSE, dendrogram='none', trace='none', key=FALSE,lwid = c(.01,.99),lhei = c(.01,.99),margins = c(5,15 ))
heatmap.2(as.matrix(ecms_normalised), Rowv=FALSE, Colv=FALSE, dendrogram='none', trace='none', key=FALSE,lwid = c(.01,.99),lhei = c(.01,.99),margins = c(5,15 ))
ggcorrplot(round(cor(ecms_normalised), 1))


# N_int <- N[,internal_reactions]
# objective <- c(rep(0, ncol(N_int)))
# objective[13] <- 1 # Biomass reaction
# target <- closest_ecm_calc_normalised
# #target <- ecms_normalised[100,]

## lpsolve only allows variables >= 0
# fluxes <- lp('max', objective, as.matrix(rbind(N_int, N_int)), rep(c('>=', '<='), c(dim(N_int)[1], dim(N_int)[1])), unlist(cbind(target - 31, target + 31)))
# fluxes[["solution"]]
# result <- t(as.matrix(N_int) %*% fluxes[["solution"]])
# colnames(result) <- metabolite_names
# View(result)

# lpSolveAPI allows any variable bound, but returns nonsensical solution dimensions
# mod = make.lp(0, ncol(N_int))
# set.objfn(mod, objective)
# lp.control(mod, sense='max')
# for (row in c(1:nrow(N_int))) {
#   add.constraint(mod, unlist(N_int[row,]), '>=', target[,row] - 0.001)
#   # add.constraint(mod, unlist(N_int[row,]), '<=', target[,row])
# }
# set.bounds(mod, lower=rep(-1000, ncol(N_int)), upper=rep(1000, ncol(N_int)), c(1:ncol(N_int)))
# write.lp(mod, 'model.lp', type='lp')
# res <- solve(mod)
# solution <- get.primal.solution(mod)
# get.solutioncount(mod)
# get.primal.solution(mod)
# View(solution)

N_int <- N[,internal_reactions]

is_ecm_feasible <- function(target) {
  # We make an irreversible stoichiometry N_irrev to use the normal lpsolve (which supports only variables >= 0)
  reversible_indices <- c(0, 1, 2, 3, 4, 5, 6, 8, 9, 11, 13, 16, 17, 18, 19, 22, 25, 27, 28, 32, 35, 37, 38, 40, 43, 48, 49, 53, 54, 56, 57, 61, 63, 64, 65, 69, 70, 72, 73, 74)+1
  N_irrev <- N_int
  for (i in reversible_indices) {
    N_irrev = cbind(N_irrev, -N_irrev[,i])
  }
  objective <- c(rep(0, ncol(N_irrev)))
  objective[13] <- 1 # Biomass reaction
  
  # We run the LP solver on our irreversible system
  fluxes <- lp('max', objective, as.matrix(rbind(N_irrev, N_irrev)), rep(c('>=', '<='), c(nrow(N_irrev), nrow(N_irrev))), unlist(cbind(target - 0.00001, target + 0.00001)))
  
  if(fluxes[['status']] > 0) {
    print('No feasible solution')
  } else {
    print(paste('Found solution with objective value', fluxes[['objval']]))
  }
  
  # And converge the resulting fluxes back into the reversible dimensions
  solution <- fluxes[["solution"]]
  for (i in seq_along(reversible_indices)) {
    solution[reversible_indices[i]] <- solution[reversible_indices[i]] + solution[ncol(N_int) + i]
  }
  solution <- as.data.frame(t(solution[c(1:ncol(N_int))]))
  colnames(solution) <- reaction_names[internal_reactions]
  solution_calc_ecm <- as.matrix(N_int) %*% t(as.matrix(solution))
  c(solution, solution_calc_ecm)
}

result <- is_ecm_feasible(closest_ecm_calc)
result <- is_ecm_feasible(ecms_diff[10,])
for (row in c(1:nrow(ecms_normalised))) {
  result <- is_ecm_feasible(ecms_normalised[row,])
}
for (row in c(1:nrow(ecms_calculated))) {
  result <- is_ecm_feasible(ecms_calculated[row,])
}

diffs <- apply(ecms_normalised, 1, function(row) { min(as.matrix(pdist(ecms_calculated, row))) })

for (row in dim(ecms_normalised)[1]) {
  print(min(as.matrix(pdist(ecms_calculated, ecms_normalised[row,]))))
}

heatmap.2(ecms_normalised, Rowv=FALSE, Colv=FALSE, dendrogram='none', trace='none', key=FALSE,lwid = c(.01,.99),lhei = c(.01,.99),margins = c(5,15 ))
