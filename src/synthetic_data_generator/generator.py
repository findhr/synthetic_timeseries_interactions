import ast
import click
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import os

from datetime import datetime, timezone, timedelta
from random import uniform, choice, randint


def do_expansion(candidates_df, action_model, shortlisted_candidates):
    prediction_results = action_model.predict(
        candidates_df[['mean_fitness_value', 'gender_', 
                       'continent_', 'candidate_ranking_']]
    )
    # select the candidate with the highest probability to be expanded if
    # she wasn't shortlisted already otherwise the next candidate in
    # the list is selected
    candidates_df['prediction'] = prediction_results
    candidates_df = candidates_df.sort_values('prediction', ascending=False)
    idx_selection = 0
    while True:
        candidate_to_expand = candidates_df.iloc[idx_selection]
        # check if the selected candidate wasn't already shortlisted
        if candidate_to_expand.candidate_id not in shortlisted_candidates:
            # if the candidate wasn't already shortlisted pick this candidate
            # to be expanded
            break
        # if the candidate was already shortlisted try with the next candidate
        # in the list
        idx_selection += 1
    return candidate_to_expand.candidate_id


def do_shortlist(candidates_df, shortlist_model, order):
    prediction_results = shortlist_model.predict(
        candidates_df[['mean_fitness_value', 'gender_', 
                       'continent_', 'candidate_ranking_']]
    )
    candidates_df['prediction'] = prediction_results
    # select the candidate with the highest probability to be shortlisted 
    # observing the selection order, meaning, for the first candidate to be 
    # shortlisted the first in the list is selected, for the second candidate
    # to be shortlisted the second candidate is selected and so on.
    candidate_to_pick = candidates_df.\
        sort_values('prediction', ascending=False).iloc[order]
    return candidate_to_pick.candidate_id
    
    
def choose_rand_expansion_duration_and_ts(job_id, condition, expansion_idx, 
                                          interactions_df):
    # select the duration of the expansion by randomly selecting among the 
    # durations of the expansions that occurred for the given condition and job.
    try:
        action_duration = choice(list(interactions_df.loc[
            (interactions_df.job_id==job_id)&\
            (interactions_df.condition==condition)&\
            (interactions_df.action_duration_sec.notna()), 
            'action_duration_sec'
        ].unique()))
        # timestamp is selected from the experimentation data considering the 
        # order in which the action occurs, meaning the first timestamp is selected if
        # it is the first expansion created in the sequence, the second if it is 
        # the second expansion created in the sequence and so on
        ts_starts = list(interactions_df.loc[
            (interactions_df.job_id==job_id)&\
            (interactions_df.condition==condition)&\
            (interactions_df.action_duration_sec.notna())
        ].sort_values('ts_start', ascending=True).ts_start)
        if len(ts_starts) > expansion_idx:
            action_ts = ts_starts[expansion_idx]
        else:
            action_ts = ts_starts[-1] + (expansion_idx - len(ts_starts)) + 1
        return {'action_duration': action_duration, 'action_ts': action_ts}
    except Exception as e:
        # if there isn't expasion activity for the given job and condition
        # return the median duration of expansions based on the experimentation data 
        action_duration = interactions_df.action_duration_sec.median()
        # if there isn't expasion activity for the given job and condition
        # return timestamp from shortlisting activity
        ts_starts = list(interactions_df.loc[
            (interactions_df.job_id==job_id)&\
            (interactions_df.condition==condition)&\
            (interactions_df.action=='shortlist')
        ].sort_values('ts_start', ascending=True).ts_start)
        if len(ts_starts) > expansion_idx:
            action_ts = ts_starts[expansion_idx]
        else:
            action_ts = ts_starts[-1] + (10 * expansion_idx)
        return {'action_duration': action_duration, 'action_ts': action_ts}


def choose_rand_shortlisting_ts(job_id, condition, num_shortlistings, 
                                interactions_df):
    shortlist_ts = list(interactions_df.loc[
            (interactions_df.job_id==job_id)&\
            (interactions_df.condition==condition)&\
            (interactions_df.action=='shortlist')
    ].sort_values('ts_start', ascending=True).ts_start) 
    return shortlist_ts[num_shortlistings]


def select_generation_parameters(data_df, prob_shortlisted_df, prob_expanded_df,
                                 prob_expansion_df, prob_shortlist_df):
    """Select generation parameters. Selection starts by randomly choosing a
    job. Based on the select job, a random condition is chosen. The job and
    condition determines the shortlist and expansion models to be used to pick
    a candidate to shortlist and expand, respectively. Finally, the probabilities
    of expanding and shortlisting are obtained. Selected parameters are stored
    in a dictionary.

    Args:
        data_df (pandas dataframe): data with the experiment results
        prob_shortlisted_df (pandas dataframe): dataframe with the logistic 
        regression models that determine the candidate to be shortlisted based on
        condition and job
        prob_expanded_df (pandas dataframe): dataframe with the logistic regression
        models that determine the candidate to be expanded based on condition and
        job
        prob_expansion_df (pandas dataframe): dataframe with the probabilities 
        of expansion based on condition and job
        prob_shortlist_df (pandas dataframe): dataframe with the probabilities of
        shortlisting based on condition and job

    Returns:
        dictionary: selected parameters
    """
    ok_models = False
    param_dict = {}
    while not ok_models:
        jobs = list(data_df.job_id.unique())
        # select random job
        rand_job = choice(jobs)
        param_dict['job'] = {'id': rand_job}
        job_dict = data_df.loc[
            data_df.job_id==rand_job,
            ['job_title', 'education_reqs_degree', 'education_reqs_major', 
             'experience_reqs_role', 'experience_reqs_duration', 
             'skills_reqs_hard', 'skills_reqs_soft']
        ].replace({np.NaN: 'none'}).iloc[0].to_dict()
        param_dict['job'].update(job_dict)
        # select random condition based on the selected random job
        rand_condition = choice(list(data_df.loc[
            data_df.job_id==rand_job, 'condition'].unique())
        )
        condition_dict = data_df.loc[
            data_df.condition==rand_condition,
            ['ranking', 'cultural_fit', 'display_name', 'display_image']
        ].replace({np.NaN: 'none'}).iloc[0].to_dict()
        param_dict['condition'] = condition_dict
        # select model to infer the probability of a candidate of being shortlisted
        ranking = condition_dict['ranking']
        priming = condition_dict['cultural_fit']
        if not condition_dict['display_image']:
            display = condition_dict['display_name']
        else:
            display = condition_dict['display_name'] + '_' + 'icon'
        # select model
        shortlist_model_df = prob_shortlisted_df.loc[
            (prob_shortlisted_df.ranking==ranking)&\
            (prob_shortlisted_df.display==display)&\
            (prob_shortlisted_df.priming==priming)
        ]
        if shortlist_model_df.shape[0] > 0:
            model_fp = shortlist_model_df.model_fname.values[0]
            # load shortlist model
            shortlist_model = sm.load(model_fp)
            # select model to infer the probability of a candidate of being expanded
            expand_model_df = prob_expanded_df.loc[
                (prob_expanded_df.ranking==ranking)&\
                (prob_expanded_df.display==display)&\
                (prob_expanded_df.priming==priming)
            ]
            if expand_model_df.shape[0] > 0:
                ok_models = True
                model_fp = expand_model_df.model_fname.values[0]
                # load expansion model
                expansion_model = sm.load(model_fp)
                # select candidates based on the random condition and job
                job_cond_candidates_df = data_df.loc[
                    (data_df.job_id==rand_job)&\
                    (data_df.condition==rand_condition)
                ].groupby('worker_id')
                job_cond_candidates_df = job_cond_candidates_df.get_group(
                    list(job_cond_candidates_df.groups.keys())[0]).reset_index()
                job_cond_candidates_df = \
                    job_cond_candidates_df[['candidate_id', 'mean_fitness_value', 'gender_', 
                                            'continent_', 'candidate_ranking_']]
                sub_data_df = data_df[(data_df.job_id==rand_job)&(data_df.condition==rand_condition)].groupby('worker_id')
                job_cond_candidates_df = sub_data_df.get_group(list(sub_data_df.groups.keys())[0]).reset_index()
                job_cond_candidates_df = job_cond_candidates_df.replace({np.NaN: 'none'})
                param_dict['candidates'] = job_cond_candidates_df[
                    ['candidate_id', 'name', 'gender', 'continent', 'unusual', 
                    'candidate_edu', 'candidate_exp', 'skills', 'candidate_ranking'
                    ]
                ].to_dict('records')
                # select probability of expansion based on the random job and 
                # condition
                prob_expansion = \
                    prob_expansion_df.loc[
                        (prob_expansion_df.condition==rand_condition)&\
                        (prob_expansion_df.job_id==rand_job),
                        'prob_expansion'
                    ].values[0]
                # select probability of shortlisted after expansion based on the 
                # random job and condition
                prob_shortlist_after_expansion = \
                    prob_shortlist_df.loc[
                        (prob_expansion_df.condition==rand_condition)&\
                        (prob_expansion_df.job_id==rand_job),
                        'prob_shortlist_after_expansion'
                    ].values[0]
    return {
        'rand_job': rand_job, 'rand_condition': rand_condition, 
        'candidates': job_cond_candidates_df, 'param_dict': param_dict, 
        'prob_expansion': prob_expansion, 'shortlist_model': shortlist_model,
        'prob_shortlist_after_expansion': prob_shortlist_after_expansion,
        'expansion_model': expansion_model
    }


def generate_synthetic_data(data_df, prob_expansion_df, prob_shortlist_df,
                            prob_shortlisted_df, prob_expanded_df, 
                            interactions_df, num_records=100):
    generation_report = {}
    interactions = []
    for i in range(num_records):
        gen_parameter_dict = {
            'num_record': i+1
        }
        selected_parameters = \
            select_generation_parameters(data_df, prob_shortlisted_df, 
                                         prob_expanded_df, prob_expansion_df,
                                         prob_shortlist_df)
        gen_parameter_dict.update(selected_parameters['param_dict'])
        rand_job = selected_parameters['rand_job']
        rand_job_title = data_df.loc[data_df.job_id==rand_job, 'job_title'].iloc[0]
        rand_condition = selected_parameters['rand_condition']
        job_cond_candidates_df = selected_parameters['candidates']
        shortlist_model = selected_parameters['shortlist_model']
        expansion_model = selected_parameters['expansion_model']
        prob_expansion = selected_parameters['prob_expansion']
        prob_shortlist_after_expansion = selected_parameters['prob_shortlist_after_expansion']
        print(f'[{i+1}/{num_records}] Generating interactions for job: {rand_job_title} and condition: {rand_condition}')
        key_report = f'{rand_job_title}#{rand_condition}'
        if key_report in generation_report:
            generation_report[key_report]['freq'] += 1
        else:
            generation_report[key_report] = {
                'freq': 1,
                'params': selected_parameters['param_dict']['condition']
            }
        # iterate until three candidates are selected
        shortlisted_candidates = []
        gen_parameter_dict['interactions'] = []
        count_expansions = 0
        while len(shortlisted_candidates) < 3:
            do_a_expansion = uniform(0,1.1)
            if do_a_expansion >= prob_expansion:
                # do expansion
                expanded_candidate = \
                    do_expansion(job_cond_candidates_df, expansion_model, 
                                 shortlisted_candidates)
                expansion_dict = \
                    choose_rand_expansion_duration_and_ts(
                        rand_job, rand_condition, count_expansions, 
                        interactions_df)
                gen_parameter_dict['interactions'].append(
                    {
                        'candidate_id': expanded_candidate,
                        'action': 'expansion',
                        'timestamp': expansion_dict['action_ts'],
                        'action_duration_sec': expansion_dict['action_duration']
                    }
                )
                count_expansions += 1
                do_a_shortlist = uniform(0,1.1)
                if do_a_shortlist >= prob_shortlist_after_expansion:
                    # do shortlist
                    shortlisted_candidate = \
                        do_shortlist(job_cond_candidates_df, shortlist_model, 
                                     len(shortlisted_candidates))
                    shortlist_ts = choose_rand_shortlisting_ts(
                        rand_job, rand_condition, len(shortlisted_candidates), 
                        interactions_df)
                    gen_parameter_dict['interactions'].append(
                        {
                            'candidate_id': shortlisted_candidate,
                            'action': 'shortlist',
                            'timestamp': shortlist_ts,
                            'action_duration_sec': ''
                        }
                    )
                    shortlisted_candidates.append(shortlisted_candidate)                    
            else:
                # do shortlist
                shortlisted_candidate = \
                        do_shortlist(job_cond_candidates_df, shortlist_model, 
                                     len(shortlisted_candidates))
                shortlist_ts = choose_rand_shortlisting_ts(
                        rand_job, rand_condition, len(shortlisted_candidates), 
                        interactions_df)
                gen_parameter_dict['interactions'].append(
                    {
                        'candidate_id': shortlisted_candidate,
                        'action': 'shortlist',
                        'timestamp': shortlist_ts,
                        'action_duration_sec': ''
                    }
                )
                shortlisted_candidates.append(shortlisted_candidate)
        gen_parameter_dict['candidated_shortlisted'] = shortlisted_candidates
        interactions.append(gen_parameter_dict)
    return interactions, generation_report
                

def prepare_data_for_regression(data_df):
    # convert gender categories to numerical values
    # out of all job description the following were identified as having more 
    # female shares. 
    # based on https://ec.europa.eu/eurostat/web/products-eurostat-news/w/edn-20230308-1
    female_jobs = ['staff nurse', 'customer service representative', 
                   'store associate', 'administrative assistant']
    data_df['gender_'] = np.where(
        data_df['gender'].isin(female_jobs),
        np.where(data_df['gender']=='female', 1, 0),
        np.where(data_df['gender']=='male', 1, 0)
    )
    # convert continent categories to numerical values
    data_df['continent_'] = np.where(data_df['continent']=='EU', 1, 0)
    # turn candidate rankings upside down
    data_df['candidate_ranking_'] = 9 - data_df['candidate_ranking']
    # convert shortlisted categories to numerical values
    data_df['shortlisted_'] = np.where(data_df['shortlisted']==True, 1, 0)
    return data_df


def compute_prob_beign_expanded(data_df, output_dir):
    prob_expanded_fp = os.path.join(output_dir, 'prob_expanded_conditions.csv')
    if os.path.isfile(prob_expanded_fp):
        print('Found dataset containing the probability of beign expanded, skipping generation...')
        prob_expanded_df = pd.read_csv(prob_expanded_fp)
    else:
        print('Generating dataset containing the probability of beign expanded...')
        # create target variable
        data_df['expanded_'] = np.where(data_df['n_views']>0,1,0)
        # select interested columns
        sub_data_df = data_df[['worker_id', 'job_id', 'candidate_id', 
                               'condition', 'ranking', 'display', 'priming',
                               'mean_fitness_value', 'gender_', 
                               'continent_', 'candidate_ranking_', 
                               'expanded_']]
        # fit a logistic regression (LR) model per each condition
        conditions = sub_data_df.condition.unique()
        prob_expanded = []
        model_dir = os.path.join(output_dir, 'models')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        for condition in conditions:
            print(f'Condition: {condition}')
            condition_df = sub_data_df[sub_data_df.condition==condition]
            features = condition_df[['mean_fitness_value', 'gender_', 'continent_', 'candidate_ranking_']]
            target = condition_df[['expanded_']]
            try:
                lr_model = sm.Logit(target, features).fit()
                lr_model_fn = f'lrmodel_{condition}_expanded.pkl'
                lrmodel_fp = os.path.join(model_dir, lr_model_fn)
                lr_model.save(lrmodel_fp)
                model_dict = {
                    'ranking': condition_df['ranking'].values[0],
                    'display': condition_df['display'].values[0],
                    'priming': condition_df['priming'].values[0],
                    'model_fname': lrmodel_fp,
                    'r_squared': lr_model.prsquared
                }
                # add coefficients
                model_dict.update({f'coef_{k}': v for k, v in lr_model.params.to_dict().items()})
                # add p_values
                model_dict.update({f'pvalue_{k}': v for k, v in lr_model.pvalues.to_dict().items()})
                # add standard errors
                model_dict.update({f'sterror_{k}': v for k, v in lr_model.bse.to_dict().items()})
                prob_expanded.append(model_dict)
            except:
                pass
        prob_expanded_df = pd.DataFrame(prob_expanded)
        prob_expanded_df.to_csv(prob_expanded_fp, index=False)
    return prob_expanded_df


def compute_prob_beign_shortlisted(data_df, output_dir):
    prob_shortlisted_conditions_fp = os.path.join(output_dir, 'prob_shortlisted_conditions.csv')
    if os.path.isfile(prob_shortlisted_conditions_fp):
        print('Found dataset containing the probability of beign shortlisted, skipping generation...')
        prob_shortlisted_df = pd.read_csv(prob_shortlisted_conditions_fp)
    else:
        print('Generating the dataset containing the probability of beign shortlisted...')
        data_df = prepare_data_for_regression(data_df)
        # select interested columns
        sub_data_df = data_df[['worker_id', 'job_id', 'candidate_id', 
                               'condition', 'ranking', 'display', 'priming',
                               'mean_fitness_value', 'gender_', 
                               'continent_', 'candidate_ranking_', 
                               'shortlisted_']]
        # fit a logistic regression (LR) model per each condition
        conditions = sub_data_df.condition.unique()
        prob_shortlisted = []
        model_dir = os.path.join(output_dir, 'models')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        for condition in conditions:
            print(f'Condition: {condition}')
            condition_df = sub_data_df[sub_data_df.condition==condition]
            features = condition_df[['mean_fitness_value', 'gender_', 'continent_', 
                                     'candidate_ranking_']]
            target = condition_df[['shortlisted_']]
            lr_model = sm.Logit(target, features).fit()
            lr_model_fn = f'lrmodel_{condition}_shortlisted.pkl'
            # save model
            lrmodel_fp = os.path.join(model_dir, lr_model_fn)
            lr_model.save(lrmodel_fp)
            model_dict = {
                'ranking': condition_df['ranking'].values[0],
                'display': condition_df['display'].values[0],
                'priming': condition_df['priming'].values[0],
                'model_fname': lrmodel_fp,
                'r_squared': lr_model.prsquared
            }
            # add coefficients
            model_dict.update({f'coef_{k}': v for k, v in lr_model.params.to_dict().items()})
            # add p_values
            model_dict.update({f'pvalue_{k}': v for k, v in lr_model.pvalues.to_dict().items()})
            # add standard errors
            model_dict.update({f'sterror_{k}': v for k, v in lr_model.bse.to_dict().items()})
            prob_shortlisted.append(model_dict)
        prob_shortlisted_df = pd.DataFrame(prob_shortlisted)
        prob_shortlisted_df.to_csv(prob_shortlisted_conditions_fp, index=False)
    return prob_shortlisted_df
    

def compute_prob_shortlist_after_expansion(interactions_df, output_dir):
    prob_shortlist_fp = os.path.join(output_dir, 'prob_shortlist_after_expansion.csv')
    if os.path.isfile(prob_shortlist_fp):
        print('Found dataset containing the probability of shortlisting after an expansion, skipping generation...')
        prob_shortlist_df = pd.read_csv(prob_shortlist_fp)
    else:
        print('Generating dataset containing the probability of shortlisting after an expansion...')
        job_ids = list(interactions_df.job_id.unique())
        conditions = list(interactions_df.condition.unique())
        prob_shortlist = []
        for condition in conditions:
            for job_id in job_ids:
                sub_data_df = interactions_df[
                    (interactions_df.condition==condition)&\
                    (interactions_df.job_id==job_id)
                ].sort_values(['ts_start'])
                if sub_data_df.shape[0] > 0:
                    shortlisted_candidates = list(
                        sub_data_df[sub_data_df['action']=='shortlist']['candidate']
                    )
                    num_shortlist_after_expansion = 0
                    for candidate in shortlisted_candidates:
                        expansions_df = sub_data_df[
                            (sub_data_df.candidate==candidate)&\
                            (sub_data_df.action=='expansion')
                        ]
                        if expansions_df.shape[0] > 0:
                            first_expansion_ts = expansions_df['ts_stop'].values[0]
                            shortlist_ts = \
                                sub_data_df.loc[
                                    (sub_data_df.candidate==candidate)&\
                                    (sub_data_df.action=='shortlist'), 'ts_start'
                                ].values[0]          
                            if shortlist_ts >= first_expansion_ts:
                                num_shortlist_after_expansion += 1
                    prob_shortlist_after_expasion = \
                        num_shortlist_after_expansion/len(shortlisted_candidates)
                    prob_shortlist.append(
                        {
                            'job_id': job_id,
                            'condition': condition,
                            'prob_shortlist_after_expansion': prob_shortlist_after_expasion
                        }
                    )
        prob_shortlist_df = pd.DataFrame(prob_shortlist)
        prob_shortlist_df.to_csv(prob_shortlist_fp, index=False)
    return prob_shortlist_df


def compute_prob_expansion(interactions_df, output_dir):
    prob_expansion_fp = os.path.join(output_dir, 'prob_expansion.csv')
    if os.path.isfile(prob_expansion_fp):
        print('Found dataset that contains the probability of expansion, skipping generation...')
        prob_expansion_df = pd.read_csv(prob_expansion_fp)
    else:
        print('Generating the probability of expansion...')
        job_ids = list(interactions_df.job_id.unique())
        conditions = list(interactions_df.condition.unique())
        prob_expansions = []
        for condition in conditions:
            for job_id in job_ids:
                sub_data_df = interactions_df[
                    (interactions_df.job_id==job_id)&\
                    (interactions_df.condition==condition)
                ].sort_values('ts_start').groupby(['worker_id']).first().\
                    reset_index()[['job_id', 'condition', 'action']]
                if sub_data_df.shape[0] > 0:
                    prob_expansion = sub_data_df[sub_data_df.action=='expansion'].shape[0]/sub_data_df.shape[0]
                    prob_expansions.append(
                        {
                            'job_id': job_id,
                            'condition': condition,
                            'prob_expansion': prob_expansion
                        }
                    )
        prob_expansion_df = pd.DataFrame(prob_expansions)
        prob_expansion_df['prob_shortlist'] = 1 - prob_expansion_df['prob_expansion']
        prob_expansion_df.to_csv(prob_expansion_fp, index=False)
    return prob_expansion_df


def generate_expansion_interactions(data_df, output_dir):
    """
    Expansions for a given candidate are originally saved all together in the 
    same record. Here, they are split into as many records as expansions exist 
    for the candidate.

    Args:
        data_df (pandas dataframe): dataframe including all data generated from 
        the experiments
        output_dir (string): path to the output directory

    Returns:
        pandas dataframe: dataframe including expansion interactions
    """
    actions = []
    for _, row in data_df.iterrows():
        action = row.to_dict().copy()
        num_actions = action['n_views']
        if num_actions > 0:
            timestamps = action['timestamps'].split(',')
            action.pop('timestamps')
            for i in range(0, num_actions+1, 2):
                try:
                    ts_start = int(timestamps[i].replace('start:','').strip())/1000
                    ts_stop = int(timestamps[i+1].replace('stop:','').strip())/1000
                    action['action'] = 'expansion'
                    action['ts_start'] = ts_start
                    action['ts_stop'] = ts_stop
                    action['action_duration_sec'] = \
                        (datetime.fromtimestamp(ts_stop, tz=timezone.utc) - \
                            datetime.fromtimestamp(ts_start, tz=timezone.utc)).seconds
                    actions.append(action.copy())
                except:
                    print(f'Discarding timestamps: {timestamps} for candidate: {action["candidate"]}')
        else:
            action.pop('timestamps')
            action['action'] = 'expansion'
            action['ts_start'] = np.NaN
            action['ts_stop'] = np.NaN
            action['action_duration_sec'] = np.NaN
            actions.append(action)
    all_expands_df = pd.DataFrame(actions)
    expand_df = all_expands_df[all_expands_df.n_views>0]
    # save to file
    expand_df.to_csv(os.path.join(output_dir, 'expansions.csv'), index=False)
    return all_expands_df


def generate_shortlist_interactions(all_expands_df, output_dir):
    conditions = list(all_expands_df.condition.unique())
    jobs = list(all_expands_df.job_id.unique())
    workers = list(all_expands_df.worker_id.unique())
    sl_actions = []
    # reference timestamp for case 1 based on the date and time of the
    # first expansion that occurred in the experiment
    ref_dt = datetime.strptime('2024-04-03 17:15:10.05', '%Y-%m-%d %H:%M:%S.%f')
    for job in jobs:
        for condition in conditions:
            for worker in workers:
                sub_data_df = all_expands_df[
                    (all_expands_df.condition==condition)&\
                    (all_expands_df.job_id==job)&\
                    (all_expands_df.worker_id==worker)&\
                    (all_expands_df.shortlisted==True)
                ]
                if sub_data_df.shape[0] > 0:
                    candidate_ids = list(sub_data_df.candidate_id.unique())
                    if sub_data_df['n_views'].sum() == 0:
                        # case 1: candidates were shortlisted but not expanded at all
                        # shortlists happen in current timestamp and shortlisting order
                        for candidate_id in candidate_ids:
                            candidate_df = sub_data_df[sub_data_df['candidate_id']==candidate_id]
                            sl_action = candidate_df.iloc[0].to_dict().copy()
                            sl_action['action'] = 'shortlist'
                            order_shortlisted = candidate_df['order_shortlisted'].values[0]
                            # add minutes in the range of -10 to 10 from the 
                            # reference timestamp to give more variability to
                            # the timestamp when shortlisting in this case occurr
                            rand_delta = randint(-10,10)
                            shortlist_dt = ref_dt + timedelta(seconds=(rand_delta*60))
                            shortlist_ts = datetime.timestamp(shortlist_dt)+int(order_shortlisted)
                            sl_action['ts_start'] = shortlist_ts
                            sl_actions.append(sl_action)
                    else:
                        for candidate_id in candidate_ids:
                            candidate_df = sub_data_df[sub_data_df['candidate_id']==candidate_id]
                            sl_action = candidate_df.iloc[0].to_dict().copy()
                            if candidate_df['n_views'].sum() == 0:
                                # case 2: candidate was shortlisted but not expanded
                                # shortlist happens before first expansion starts
                                shortlist_ts = \
                                    sub_data_df[
                                        sub_data_df.candidate!=candidate_id
                                    ].sort_values(['order_shortlisted', 'ts_start'])['ts_start'].values[0]-1
                            else:
                                # case 3: candidate was shortlisted and expanded
                                # shortlist happens after last expansion ends
                                shortlist_ts = candidate_df['ts_stop'].max()
                            sl_action['action'] = 'shortlist'
                            sl_action['ts_start'] = shortlist_ts
                            sl_action['ts_stop'] = np.NaN
                            sl_action['action_duration_sec'] = np.NaN
                            sl_actions.append(sl_action)
    shortlists_df = pd.DataFrame(sl_actions)
    shortlists_df.to_csv(os.path.join(output_dir, 'shortlists.csv'), index=False)
    return shortlists_df
    

def generate_interactions(data_df, output_dir):
    interactions_fp = os.path.join(output_dir, 'interactions.csv')
    if os.path.isfile(interactions_fp):
        print('Found interactions dataset, skipping generation...')
        interactions_df = pd.read_csv(interactions_fp)
    else:
        expansions_fp = os.path.join(output_dir, 'expansions.csv')
        if os.path.isfile(expansions_fp):
            print('Found expansion interactions dataset, skipping generation...')
            all_expands_df = pd.read_csv(expansions_fp)
        else:
            print('Generating expasion interactions...')
            all_expands_df = generate_expansion_interactions(data_df, output_dir)
        shortlists_fp = os.path.join(output_dir, 'shortlists.csv')
        if os.path.isfile(shortlists_fp):
            print('Found shortlisting interactions dataset, skipping generation...')
            shortlists_df = pd.read_csv(shortlists_fp)
        else:
            print('Generating shortlisting interactions...')
            shortlists_df = generate_shortlist_interactions(all_expands_df, output_dir)
        interactions_df = pd.concat(
            [all_expands_df[all_expands_df.n_views>0], shortlists_df], 
            ignore_index=True
        )
        interactions_df = interactions_df[
            ['job_id', 'job_title', 'ranking', 'priming', 'display', 'condition',
             'worker_id', 'action', 'ts_start', 'ts_stop', 'action_duration_sec',
             'candidate_id', 'candidate', 'gender', 'continent', 'unusual', 
             'mean_fitness_value']
        ]
        interactions_df.to_csv(interactions_fp, index=False)
    return interactions_df


def parse_candidates_order(data_df):
    job_ids = data_df.job_id.unique()
    conditions = data_df.condition.unique()
    ranking_candidates = []
    for job_id in job_ids:
        for condition in conditions:
            sub_data_df = data_df[
                (data_df['job_id']==job_id)&\
                (data_df['condition']==condition)
            ].groupby('candidate_id').first().reset_index()
            if sub_data_df.shape[0] > 0:
                for _, row in sub_data_df.iterrows():
                    candidates_ranking = ast.literal_eval(row['ordered_candidates'])
                    for idx, candidate_ranking in enumerate(candidates_ranking):
                        if candidate_ranking == row['candidate_id']:
                            candidate_order = idx
                            break
                    ranking_candidates.append(
                        {
                            'job_id': job_id,
                            'condition': condition,
                            'candidate_id': row['candidate_id'],
                            'candidate_ranking': candidate_order
                        }
                    )
    return ranking_candidates


def process_data(experiments_df, results_df, jobs_df, candidates_df, 
                 fitness_values_df, users_df, output_dir):
    # merge datasets experiments and results
    data_df = experiments_df.merge(results_df, on=['exp_id', 'task_index', 'job_id'], how='inner')
    # merge previous merging results with job descriptions
    data_df = data_df.merge(jobs_df, on=['job_id'], how='inner')
    # merge previous merging result with candidates
    data_df = data_df.merge(candidates_df, on=['candidate_id'], how='inner')
    # merge previous merging result with candidates' fitness values
    data_df = data_df.merge(fitness_values_df, on=['candidate_id'], how='inner')
    # reorder columns
    data_df = data_df.reindex(
        columns=['exp_id', 'task_index', 'prolific_id', 'job_id', 'job_title', 
                 'education_reqs_degree','education_reqs_major', 'experience_reqs_role',
                 'experience_reqs_duration', 'skills_reqs_hard', 'skills_reqs_soft',
                 'ordered_candidates', 'ranking_type', 'cultural_fit', 'display_name', 
                 'display_image', 'candidate_id', 'candidate', 'name', 'gender', 
                 'continent', 'unusual', 'candidate_edu', 'candidate_exp', 'skills', 
                 'mean_fitness_value', 'n_views', 'timestamps', 'shortlisted', 
                 'order_shortlisted'
                ]
    )
    # create new column that combines display conditions
    data_df['display'] = np.where(
        data_df['display_image'] == False, 
        data_df['display_name'], 
        data_df['display_name'] + '_' + 'icon'
    )
    # create new column that includes only ranking name
    data_df['ranking'] = data_df.ranking_type.str.replace('rank_','').str.replace('.txt','')
    # normalize cultural fit
    data_df.loc[data_df['cultural_fit']=='Employer is a large multinational in a big city in Italy.', 'cultural_fit'] = 'Employer is a large multinational in a big city'
    data_df.loc[data_df['cultural_fit']=='Employer is a large multinational in a big city in Spain.', 'cultural_fit'] = 'Employer is a large multinational in a big city'
    data_df.loc[data_df['cultural_fit']=='Employer is a large multinational in a big city in the Netherlands.', 'cultural_fit'] = 'Employer is a large multinational in a big city'
    data_df.loc[data_df['cultural_fit']=='Employer is a large multinational in a big city in Germany.', 'cultural_fit'] = 'Employer is a large multinational in a big city'
    data_df.loc[data_df['cultural_fit']=='Employer is a large multinational in a big city in France.', 'cultural_fit'] = 'Employer is a large multinational in a big city'
    data_df.loc[data_df['cultural_fit']=='Employer is a small company in a local town in Italy.', 'cultural_fit'] = 'Employer is a small company in a local town'
    data_df.loc[data_df['cultural_fit']=='Employer is a small company in a local town in Spain.', 'cultural_fit'] = 'Employer is a small company in a local town'
    data_df.loc[data_df['cultural_fit']=='Employer is a small company in a local town in the Netherlands.', 'cultural_fit'] = 'Employer is a small company in a local town'
    data_df.loc[data_df['cultural_fit']=='Employer is a small company in a local town in Germany.', 'cultural_fit'] = 'Employer is a small company in a local town'
    data_df.loc[data_df['cultural_fit']=='Employer is a small company in a local town in France.', 'cultural_fit'] = 'Employer is a small company in a local town'
    # replace NaN value in priming column by none to avoid unintentional filterings
    data_df['priming'] = np.where(data_df['cultural_fit'].isna(), 'none', data_df['cultural_fit'])
    # create new column that combines condition attributes
    data_df['condition'] = data_df['ranking'] + '_' + data_df['priming'] + '_' + data_df['display']
    # rename prolific_id column by worker_id
    data_df = data_df.rename(columns={'prolific_id':'worker_id'})
    # parse candidate orders
    candidates_order = parse_candidates_order(data_df)
    candidates_order_df = pd.DataFrame(candidates_order)
    data_df = data_df.merge(candidates_order_df, 
                            on=['job_id', 'condition', 'candidate_id'], 
                            how='inner')
    # filter out profilic workers who didn't pass the attention check
    users_ok = users_df[users_df.attention_check==True]
    data_df = data_df[data_df['worker_id'].isin(list(users_ok.prolific_id))]
    # save processed data
    data_df.to_csv(os.path.join(output_dir, 'processed_data.csv'), index=False)
    
    return data_df
    

def load_data(data_dir, output_dir):
    processed_fp = os.path.join(output_dir, 'processed_data.csv')
    if os.path.isfile(processed_fp):
        print('Found processed dataset, skipping generation...')
        data_df = pd.read_csv(processed_fp)
    else:
        print('Loading and processing experiment data...')
        # load experiment data
        experiments_df = pd.read_csv(os.path.join(data_dir, 'experiments_shortlist.csv'))
        # remove unnamed columns
        experiments_df = experiments_df.drop('Unnamed: 0', axis=1)
        # remove trailing underscore from column names
        experiments_df.columns = experiments_df.columns.str.replace('_','',1)
        
        # load results data
        results_df = pd.read_csv(os.path.join(data_dir, 'results_shortlist.csv'))
        # remove unnamed columns
        results_df = results_df.drop('Unnamed: 0', axis=1)
        # remove trailing underscore from column names
        results_df.columns = results_df.columns.str.lstrip('_')
        
        # load jobs data
        jobs_df = pd.read_csv(os.path.join(data_dir, 'jobs.csv'))
        # remove unnamed columns
        jobs_df = jobs_df.drop('Unnamed: 0', axis=1)
        # parse job ids
        jobs_df['_job_id'] = jobs_df['_id'].str.replace("{'$oid': '", '').str.replace("'}", '')
        jobs_df = jobs_df.drop('_id', axis=1)
        # rename title column
        jobs_df = jobs_df.rename(columns={'_title': '_job_title'})
        # remove trailing underscore from column names
        jobs_df.columns = jobs_df.columns.str.lstrip('_')
        
        # load candidates data
        candidates_df = pd.read_csv(os.path.join(data_dir, 'candidates.csv'))
        # remove unnamed columns
        candidates_df = candidates_df.drop('Unnamed: 0', axis=1)
        # parse candidate ids
        candidates_df['candidate_id'] = candidates_df['_id'].str.replace("{'$oid': '", '').str.replace("'}", '')
        # create new column to combine candidates' data
        candidates_df['candidate'] = candidates_df['_name'] + '_' + \
                                     candidates_df['_gender'] + '_' + \
                                     candidates_df['_continent']
        candidates_df['candidate'] = np.where(candidates_df['_unusual'] == True, 
                                     candidates_df['candidate'] + '_unusual', 
                                     candidates_df['candidate'])
        # remove trailing underscore from column names
        candidates_df.columns = candidates_df.columns.str.lstrip('_')
        
        # load candidate fitness values
        fitness_values_df = pd.read_csv(os.path.join(data_dir, 'annotation', 'DataDb', 'results_annotate.csv'))
        # remove unnamed columns
        fitness_values_df = fitness_values_df.drop('Unnamed: 0', axis=1)
        # compute average fitness value of candidates
        mean_fitness_values_df = fitness_values_df.groupby('_candidate_id')['overall_score'].mean().reset_index(name='mean_fitness_value')
        # remove trailing underscore
        mean_fitness_values_df.columns = mean_fitness_values_df.columns.str.lstrip('_')
        # load users
        users_df = pd.read_csv(os.path.join(data_dir, 'users.csv'))
        # remove unnamed columns
        users_df = users_df.drop('Unnamed: 0', axis=1)
        # remove trailing underscore
        users_df.columns = users_df.columns.str.lstrip('_')
        
        # process data
        data_df = process_data(experiments_df, results_df, jobs_df, candidates_df, 
                               mean_fitness_values_df, users_df, output_dir)
    return data_df


def save_interactions(interactions, output_dir):
    output_file_path = os.path.join(output_dir, 'interaction_timeseries.json')
    with open(output_file_path, 'w') as output_file:
        json.dump(interactions, output_file, indent=4)


def print_gen_report(report_dict, num_records, output_dir):
    report_fp = os.path.join(output_dir, 'generation_report.txt')
    with open(report_fp, 'w') as f:
        f.write(f'Interactions generated: {num_records}\n\n')
        for k, v in report_dict.items():
            job = k.split('#')[0]
            freq = v['freq']
            _ranking = v['params']['ranking']
            priming = v['params']['cultural_fit']
            if not v['params']['display_image']:
                _display = v['params']['display_name']
            else:
                _display = v['params']['display_name'] + '_' + 'icon'
            if _ranking == 'rand':
                ranking = 'random'
            elif _ranking == 'discr':
                ranking = 'discriminatory'
            elif _ranking == 'fit':
                ranking = 'fitness'
            elif _ranking == 'anti_discr':
                ranking = 'anti-discriminatory'
            priming = priming.replace('.','')
            if _display == 'initials':
                display = 'name initials'
            elif _display == 'full_icon':
                display = 'full name and icon'
            f.write(f'* Job: {job} with {ranking} ranking, {display} display, and '\
                    f'{priming} cultural fit\n')
            f.write(f'--- # interactions: {freq}\n\n')


@click.command()
@click.option('--num_records', '-nr', default=100)
def main(num_records):
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(root_dir, 'exp_data')
    output_dir = os.path.join(root_dir, 'gen_output')
    data_df = load_data(data_dir, output_dir)
    interactions_df = generate_interactions(data_df, output_dir)
    prob_expansion_df = compute_prob_expansion(interactions_df, output_dir)
    prob_shortlist_df = compute_prob_shortlist_after_expansion(interactions_df, 
                                                               output_dir)
    prob_shortlisted_df = compute_prob_beign_shortlisted(data_df, output_dir)
    prob_expanded_df = compute_prob_beign_expanded(data_df, output_dir)
    synthetic_interactions, report_dict = \
        generate_synthetic_data(data_df, prob_expansion_df, prob_shortlist_df,
                                prob_shortlisted_df, prob_expanded_df,
                                interactions_df, num_records=num_records)
    save_interactions(synthetic_interactions, output_dir)
    print_gen_report(report_dict, num_records, output_dir)


if __name__ == "__main__":
    main()





