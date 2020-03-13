cd 'C:\Users\Daan\Documents\GitHub\ecmtool\results_analysis\tmp'

N = table2array(readtable('ex_N.csv','ReadVariableNames',false));
reversibilities = table2array(readtable('reversibilities.csv','ReadVariableNames',false));

cone_table = readtable('conversion_cone_KO1.csv');
cone = table2array(cone_table)';
fid = fopen('metab_ids.csv');
metab_ids = textscan(fid, '%s', 'Delimiter',',');
fclose(fid);
metab_ids = metab_ids{1};

fid = fopen('reac_ids.csv');
reac_ids = textscan(fid, '%s', 'Delimiter',',');
fclose(fid);
reac_ids = reac_ids{1};

efmtool_folder = 'C:\\Users\\Daan\\surfdrive\\PhD\\Software\\efmtool';
cd(efmtool_folder);

opts = CreateFluxModeOpts('arithmetic','fractional','precision',-1,'normalization','none');
mnet = CalculateFluxModes(N, reversibilities,opts);

efms =  mnet.efms;

n_exchange=11;
efms_crop = efms(1:end-n_exchange,:);
N_crop=N(:,1:end-n_exchange);
efmecms=N_crop*efms_crop;

% Normalize efmecms
norm_factor = sum(abs(efmecms),1);
norm_factor(norm_factor == 0) = 1;
efms = efms ./ repmat(norm_factor,length(efms(:,1)),1);
efmecms = efmecms./repmat(norm_factor,length(efmecms(:,1)),1);

% Normalize ecms
norm_factor = sum(abs(cone),1);
norm_factor(norm_factor == 0) = 1;
cone = cone./repmat(norm_factor,length(cone(:,1)),1);

% Find the external metabolite rows from the efmecms and order them as the
% cone is ordered
ext_efmecms = [];
external_inds = [];
for i=1:numel(cone_table.Properties.VariableNames)
    variable_name = cone_table.Properties.VariableNames{i};
    ind = find(strcmp(metab_ids,variable_name));
    external_inds = [external_inds,ind];
    ext_efmecms = [ext_efmecms;efmecms(ind,:)];
end
internal_inds = setxor(1:numel(metab_ids),external_inds);
max_internal_production = max(efmecms(internal_inds,:));

% Loop over ecms, and see how many efmecms map to these
clusters = {};
tol = 1e-12;
clustered_efmecms = [];
for i=1:length(cone(1,:))
    ecm = cone(:,i);
    deviations = sum(abs(ext_efmecms - repmat(ecm,1,length(ext_efmecms(1,:)))),1);
    similar_ecm_inds = find(deviations<tol);
    clusters{i} = similar_ecm_inds;
    clustered_efmecms = [clustered_efmecms,similar_ecm_inds];
end
unclustered = setxor(1:length(ext_efmecms(1,:)),clustered_efmecms);

% We pick the ecm that is not clustered and produces maximal biomass
% The followin analysis will be specific for this one
[~,sorted_ind] = sort(ext_efmecms(11,unclustered));
weird_inds=unclustered(sorted_ind);
weird_ind = weird_inds(3);

% Try to find which ecms give rise to this unclustered ecm
n_ecms = length(cone(1,:));
f = ones(n_ecms,1);
A = [];
b = [];
Aeq = cone;
beq = ext_efmecms(:,weird_ind);
LB = zeros(n_ecms,1);
UB = Inf*ones(n_ecms,1);
[X,FVAL,~,OUTPUT] = linprog(f,A,b,Aeq,beq,LB,UB);

% So, we have the combination of ecms in X giving the same conversion as
% our weird efmecm. We now find the efms that belong in this combination
X_efms = zeros(1,length(efms(1,:)));
for i=1:length(X)
    if X(i)>0
        X_efms(clusters{i}) = X(i);
    end
end

% Put these EFMs together in an array, scaled appropriately 
compare_efms = [efms(:,find(X_efms>0)),efms(:,weird_ind)];
compare_efms = compare_efms./repmat(compare_efms(end,:),length(compare_efms(:,1)),1);
compare_efms_scaled = [efms(:,find(X_efms>0)),efms(:,weird_ind)].*[X_efms(X_efms>0),1];

% Find reactions that are changed in terms of sign +/0/-
changed_reacs = [];
for i=1:length(compare_efms_scaled(:,1))
    if numel(unique(sign(compare_efms_scaled(i,:))))>1
        changed_reacs = [changed_reacs,i];
    end
end

summed_reacs_ecms_with_weird_one =[sum(compare_efms_scaled(:,1:sum(X_efms>0)),2),compare_efms_scaled(:,end)];
compare_efms_scaled_table = array2table(compare_efms_scaled,'RowNames',reac_ids);
compare_efms_table = array2table(compare_efms,'RowNames',reac_ids);
compare_ecms_combined_weirdone = array2table(summed_reacs_ecms_with_weird_one,'RowNames',reac_ids);
compare_efms_table(changed_reacs,:)

% The below tries to identify different efms, but doesn't identify much
clusters = {};
tol = 1e-6;
counter = 1;
clustered_efmecms = 0;
efmecms_complete = efmecms;
while ~isempty(efmecms)
    counter
    if counter>100
        sprintf('More than 100 clusters needed, %d ecms clustered',clustered_efmecms)
        break
    end
    ecm = efmecms(:,1);
    deviations = sum(abs(efmecms - repmat(ecm,1,length(efmecms(1,:)))),1);
    similar_ecm_inds = find(deviations<tol);
    clusters{counter} = similar_ecm_inds;
    efmecms(:,similar_ecm_inds)=[];
    counter = counter + 1;
    clustered_efmecms = clustered_efmecms + length(similar_ecm_inds);
end