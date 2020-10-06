### Description models and runscripts

In this folder, one can find the calculated sets of ECMs with the run arguments that were used in the command line.

#### `e_coli_core`

The `e_coli_core` model was downloaded as an .xml-file from `bigg.ucsd.edu`. For this model we show a couple different methods

- Full calculation of all ECMs

- Calculation of ECMs while hiding all metabolites that can be excreted

- Calculation of ECMs while hiding all metabolites that can be excreted, while tagging the use of the PDH-reaction


#### `iIT341`

This model of *Helicobacter pylori* was downloaded as an .xml-file from `bigg.ucsd.edu`. For this model, we computed the ECMs on the minimal medium MinII indicated in the paper by Thiele et al. (https://doi.org/10.1128/jb.187.16.5818-5830.2005). All medium compounds except for H2O could only be used as a substrate, H2O could be used as both a substrate as a product. We calculated

- All ECMs, but hiding information about H2O uptake/production
- ECMs while hiding all information about produced compounds

Furthermore, we calculated the ECMs with hidden outputs on a medium where H2O was not included. This was just to check if *H. pylori* can grow without taking up H2O.

#### `iJR904`

This model of *E. coli* was downloaded as an .xml-file from `bigg.ucsd.edu`. We calculated the ECMs for this model, but discarding all information about the external metabolites that are not glucose, oxygen of the biomass metabolite.

  

