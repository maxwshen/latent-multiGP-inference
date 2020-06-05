# 
import sys, string, pickle, subprocess, os, datetime, gzip, time
import numpy as np, pandas as pd
import scipy
from collections import defaultdict
import util
import _config, _data

import torch, torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils
import torch.nn as nn
import glob
import gpytorch
import pyro

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

#
NAME = util.get_fn(__file__)
data_dir = _config.OUT_PLACE + f'b_prep_data_v2/'
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)
log_fn = None
fold_nm = ''

paper_features_df = pd.read_csv(data_dir + 'paper_features.csv', index_col = 0, parse_dates = ['Earliest order date', 'Publication date'])

#
hparams = {
  'num_epochs': 5000,
  'num_batches': 100,
  'learning_rate': .0001,
  # 'learning_rate': 0.05,
  'weight_decay': 1e-5,
  'momentum': 0.9,

  # learning rate scheduler
  'plateau_patience': 5,
  'plateau_threshold': 1e-4,
  'plateau_factor': 0.1,

  'random_seed': 0,

  'dataset': 'US',

  'num_inducing': 512,
  'subsample_size': 1000,
}

##
# Support
##
def parse_custom_hyperparams(custom_hyperparams):
  # Defaults
  if custom_hyperparams == '':
    return

  parse_funcs = {
    'slash-separated list': lambda arg: [int(s) for s in arg.split('/')],
    'binary': lambda arg: bool(int(arg)),
    'int': lambda arg: int(arg),
    'float': lambda arg: float(arg),
    'str': lambda arg: str(arg),
  }

  term_to_parse = {
    'num_epochs': parse_funcs['int'],
    'learning_rate': parse_funcs['float'],
    'batch_size': parse_funcs['int'],
    'num_batches': parse_funcs['int'],
    'momentum': parse_funcs['float'],
    'weight_decay': parse_funcs['float'],

    # learning rate schedulers
    'plateau_patience': parse_funcs['int'],
    'plateau_factor': parse_funcs['float'],

    # loss
    'loss_func': parse_funcs['str'],

    # architecture
    'hidden_sizes': parse_funcs['slash-separated list'],
    'dropout_p': parse_funcs['float'],

    'dataset': parse_funcs['str'],
    'random_seed': parse_funcs['int'],
    'mle_init': parse_funcs['binary'],
  }

  # Parse hyperparams
  global hparams
  for term in custom_hyperparams.split('+'):
    [kw, args] = term.split(':')
    if kw in hparams:
      parse = term_to_parse[kw]
      hparams[kw] = parse(args)

  return

def copy_model_script():
  from shutil import copyfile
  copyfile(__file__, model_dir + f'{NAME}.py')

def check_num_models():
  import glob
  dirs = glob.glob(out_dir + 'model*')
  return len(dirs)

def print_and_log(text):
  with open(log_fn, 'a') as f:
    f.write(text + '\n')
  print(text)
  return

def create_model_dir():
  num_existing = check_num_models()
  global model_dir
  if fold_nm == '':
    run_id = str(num_existing + 1)
  else:
    run_id = fold_nm

  model_dir = out_dir + 'model_' + run_id + '/'
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  subprocess.check_output(f'rm -rf {model_dir}*', shell = True)
  print('Saving model in ' + model_dir)

  global log_fn
  log_fn = out_dir + '_log_%s.out' % (run_id)
  with open(log_fn, 'w') as f:
    pass
  print_and_log('model dir: ' + model_dir)
  return

def set_random_seed():
  rs = hparams['random_seed']
  print(f'Using random seed: {rs}')
  np.random.seed(seed = rs)
  torch.manual_seed(rs)
  return


##
# Models
##
class LowLevelGPModel(gpytorch.models.ApproximateGP):
  '''
    GPyTorch + Pyro low-level integration using: 
      https://gpytorch.readthedocs.io/en/latest/examples/07_Pyro_Integration/Pyro_GPyTorch_Low_Level.html
    and Pyro documentation to include additional latent random variables.

    Uses low-level integration with GPyTorch's ApproximateGP class: https://gpytorch.readthedocs.io/en/latest/models.html#models-for-approximate-gp-inference
  '''
  def __init__(self, package, name_prefix = 'llgm', num_inducing = hparams['num_inducing']):
    '''
    '''
    self.name_prefix = name_prefix
    self.paper_nms = package['paper_nms']
    self.factor_nms = package['factor_nms']
    self.max_date_int = package['max_date_int']

    # torch Tensor of all unique times. Used to evaluate GP
    self.times = torch.Tensor(np.arange(self.max_date_int + 1))

    self.num_factors = len(self.factor_nms)
    self.num_papers = len(self.paper_nms)

    if len(self.times) < num_inducing:
      num_inducing = len(self.times)
    print(f'Using {num_inducing} inducing points ...')
    self.num_inducing = num_inducing

    # Indexers
    self.paper_nm_to_param_idx = {pnm: idx for idx, pnm in enumerate(self.paper_nms)}
    self.param_idx_to_paper_nm = {idx: pnm for idx, pnm in enumerate(self.paper_nms)}
    self.factor_nm_to_param_idx = {fnm: idx for idx, fnm in enumerate(self.factor_nms)}
    self.param_idx_to_factor_nm = {idx: fnm for idx, fnm in enumerate(self.factor_nms)}

    '''
      Setup GPs
    '''
    # Define all the variational stuff
    variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
      num_inducing_points = self.num_inducing,
      batch_shape = torch.Size([self.num_papers])
    )

    # Here we're using a MultitaskVariationalStrategy - so that the output of the
    # GP latent function is a MultitaskMultivariateNormal
    # Inducing points lay on an evenly spaced grid from 0 to T
    inducing_points = torch.linspace(0, self.max_date_int, self.num_inducing).unsqueeze(-1)
    variational_strategy = gpytorch.variational.MultitaskVariationalStrategy(
      gpytorch.variational.VariationalStrategy(
        self, 
        inducing_points, 
        variational_distribution,
        learn_inducing_locations = True,
      ),
      num_tasks = self.num_papers,
    )

    # Standard initializtation
    super().__init__(variational_strategy)

    # Mean, covar
    self.mean_module = gpytorch.means.ZeroMean()
    self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    # self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures = 10)

    # Set up prior latent variables
    # self.factor_means = torch.nn.Parameter()
    self.factormean = -1 * torch.ones(self.num_factors)
    self.factorstd = torch.ones(self.num_factors)

  def forward(self, x):
    '''
      GP forward
    '''
    mean_x = self.mean_module(x)
    covar_x = self.covar_module(x)
    return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

  def guide(self, data):
    '''
      Get q(f) - variational (guide) distributions for each random variable
      * Must have the same arguments as self.model().
      * Must call pyro.sample for the same named random variables as self.model(), without using observations
    '''
    # Sample from guides for latent variables
    # Initialize variational guide distribution parameter values
    factormean_q = pyro.param('factormean_q', -1 * torch.ones(self.num_factors))
    factorstd_q = pyro.param('factorstd_q', torch.ones(self.num_factors), constraint = torch.distributions.constraints.positive)
    pyro.sample(self.name_prefix + '.factor_effects', pyro.distributions.LogNormal(factormean_q, factorstd_q))

    # This works better, surprisingly fast too
    function_dist = self.pyro_guide(self.times)
    pyro.sample(self.name_prefix + f'.f(x)', function_dist)
    # Sample shape is [num_times, num_gps]
    pass

  def model(self, data):
    '''
      Model with custom likelihood as a function of input data features and latent variables.

      data = list of dicts, {
        'obs_count': int,
        'paper_nm': str,
        'date': pd.Timestamp,
        'date_int': torch.Tensor([int]),
        'weekend': bool,
        'covid': bool,
        'christmas': bool,
        'num_days_since_start': int,
      }
    '''
    pyro.module(self.name_prefix + '.gp', self)

    # Sample from latent factors
    factor_sample = pyro.sample(self.name_prefix + '.factor_effects', pyro.distributions.LogNormal(self.factormean, self.factorstd))

    '''
      Sample from latent function distribution.
      Observations are independent when conditioning on GP samples and latent variable samples
    '''
    # Get p(f) - prior distribution of latent function
    function_dist = self.pyro_guide(self.times)
    gp_sample_mat = pyro.sample(self.name_prefix + f'.f(x)', function_dist)
    # gp_sample_mat shape is [num_times, num_gps]

    # Explicit sequential plate, slower than vectorized plate, but compatible with data = list of dicts (very convenient for development) and does not require Tensors
    for i in pyro.plate(self.name_prefix + '.data_plate', len(data), subsample_size = hparams['subsample_size']):
      x = data[i]
      time_idx = x['date_int']
      paper_idx = self.paper_nm_to_param_idx[x['paper_nm']]

      # Form rate
      gp_sample = gp_sample_mat[time_idx, paper_idx]
      paper_effect = torch.abs(gp_sample)
      mult_factor = 1
      for fidx, factor_nm in enumerate(self.factor_nms):
        if x[factor_nm]:
          mult_factor = mult_factor * factor_sample[fidx]

      # Sample from observed distribution
      obs_distribution = pyro.distributions.Poisson(paper_effect * mult_factor)
      pyro.sample(
        self.name_prefix + f'.y{i}',
        obs_distribution,
        obs = x['obs_count'],
      )
      # import code; code.interact(local=dict(globals(), **locals()))
    pass


def save_params(model, epoch, stats_dd, loss):
  '''
  '''
  stats_dd['Epoch'].append(epoch)
  stats_dd['Loss'].append(loss)

  ls = model.covar_module.base_kernel.lengthscale.item()
  stats_dd['GP lengthscale'].append(ls)

  get_param = lambda x: list(pyro.param(x).detach().numpy())
  factor_means = get_param('factormean_q')
  factor_stds = get_param('factorstd_q')

  for fidx, factor_nm in enumerate(model.factor_nms):
    m = factor_means[fidx]
    s = factor_stds[fidx]
    stats_dd[f'{factor_nm} lognormal mean'].append(m)
    stats_dd[f'{factor_nm} lognormal std'].append(s)
    stats_dd[f'{factor_nm} mean'].append(np.exp(m))
    stats_dd[f'{factor_nm} 2.5 pct'].append(np.exp(m - 2*s))
    stats_dd[f'{factor_nm} 97.5 pct'].append(np.exp(m + 2*s))

  stats_df = pd.DataFrame(stats_dd)
  stats_df.to_csv(model_dir + f'inferred_parameters.csv')
  return


##
# Training
##
def train_model(model, optimizer, data):
  stats_dd = defaultdict(list)
  since = time.time()
  model.train()

  elbo = pyro.infer.Trace_ELBO()
  svi = pyro.infer.SVI(model = model.model, guide = model.guide, optim = optimizer, loss = elbo)

  num_epochs = hparams['num_epochs']
  epoch_loss = dict()
  losses = []
  for epoch in range(num_epochs):
    print_and_log('-' * 10)
    print_and_log('Epoch %s/%s at %s' % (epoch, num_epochs - 1, datetime.datetime.now()))
    for phase in ['train']:
      model.train() if phase == 'train' else model.eval()
      # running_loss = 0.0
      with torch.set_grad_enabled(phase == 'train'):

        model.zero_grad()
        loss = svi.step(data)

        # Print stats
        batch_loss_float = loss
        print_and_log(f'{phase}: {batch_loss_float:.3E}')

      # Each phase
      # epoch_loss[phase] = running_loss / num_batches

    # Each epoch
    pass

    save_under_100 = epoch < 100 and epoch % 20 == 1
    save_over_100 = epoch >= 100 and epoch % 100 == 1
    save_criteria = save_under_100 or save_over_100
    # save_criteria = True
    save_params(model, epoch, stats_dd, batch_loss_float)
    if save_criteria:
      torch.save(model.state_dict(), model_dir + f'model_epoch_{epoch}.pt')

    # losses.append(epoch_loss)

  time_elapsed = time.time() - since
  print_and_log('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

  torch.save(model.state_dict(), model_dir + 'model_final.pt')
  return


'''
  Data
'''
def load_data(dataset_nm):
  '''
    data is a list of dicts with {
      'obs_count': int,
      'paper_nm': str,
      'date': pd.Timestamp,
      'weekend': bool,
      'covid': bool,
      'christmas': bool,
      'num_days_since_start': int,
    }

    data_stats is used to initialize the model.
  '''
  data_df = pd.read_csv(data_dir + f'{dataset_nm}_papertime_data.csv', index_col = 0, parse_dates = ['Order Date'])

  # Toy 
  crit = (data_df['Order Date'] >= pd.Timestamp('2019-01-15'))
  data_df = data_df[crit]

  earliest_date = min(data_df['Order Date'])
  data_df = data_df.set_index('Order Date')
  paper_nms = list(data_df.columns)

  # Preprocess date features
  date_features = pd.read_csv(data_dir + f'date_features.csv', index_col = 0, parse_dates = ['Date'])
  print(f'Preprocessing features for {len(date_features)} dates ...')
  datetime_d = defaultdict(dict)
  stat_cols = ['Weekend', 'Christmas holiday', 'Covid19 china', 'Covid19 us-eur']
  timer = util.Timer(total = len(date_features))
  for idx, row in date_features.iterrows():
    date = row['Date']
    for sc in stat_cols:
      datetime_d[date][sc] = row[sc]
    timer.update()

  # Preprocess paper features
  paper_features = pd.read_csv(data_dir + 'paper_features.csv', index_col = 0, parse_dates = ['Earliest order date', 'Publication date'])
  print(f'Preprocessing features for {len(paper_features)} dates ...')
  papers_d = defaultdict(dict)
  stat_cols = ['Earliest order date']
  timer = util.Timer(total = len(paper_features))
  for idx, row in paper_features.iterrows():
    doi = row['DOI']
    for sc in stat_cols:
      papers_d[doi][sc] = row[sc]
    timer.update()

  # Form data
  data = []
  timer = util.Timer(total = len(paper_nms))
  max_date_int = -1
  print(f'Forming dataset with {len(paper_nms)} papers ...')
  for paper in paper_nms:
    '''
      dfs has index = date (pd.Timestamp) and one column = DOI. Values are plasmid request counts.
    '''
    dfs = pd.DataFrame(data_df[paper].dropna())
    ndss = papers_d[paper]['Earliest order date']
    # import code; code.interact(local=dict(globals(), **locals()))
    for date, row in dfs.iterrows():
      delta = date - ndss
      num_days_since_start = delta.days
      num_days_since_t0 = (date - earliest_date).days
      if num_days_since_t0 > max_date_int:
        max_date_int = num_days_since_t0
      d = {
        'obs_count': row[paper],
        'paper_nm': paper,
        'date': date,
        'date_int': num_days_since_t0,
        'weekend': datetime_d[date]['Weekend'],
        'covid': datetime_d[date]['Covid19 us-eur'] if hparams['dataset'] != 'China' else datetime_d[date]['Covid19 china'],
        'christmas': datetime_d[date]['Christmas holiday'],
        'num_days_since_start': num_days_since_start,
      }
      data.append(d)
    timer.update()
  print(f'Found {len(data)} observations')

  # Init data stats
  data_stats = dict()
  data_stats['paper_nms'] = paper_nms
  data_stats['factor_nms'] = ['weekend', 'christmas', 'covid']
  data_stats['max_date_int'] = max_date_int

  return data, data_stats


'''
  Primary
'''
def run_inference(batch_nm = '', exp_nm = '', custom_hyperparams = ''):
  parse_custom_hyperparams(custom_hyperparams)
  set_random_seed()

  # Load data
  dataset_nm = hparams['dataset']
  print(f'Loading dataset {dataset_nm} ...')
  data, data_stats = load_data(dataset_nm)
  print(data_stats)

  # Set up environment
  global out_dir
  if batch_nm == '': batch_nm = 'unnamed'
  out_dir = out_dir + batch_nm + '/' if batch_nm != '' else out_dir
  util.ensure_dir_exists(out_dir)
  global fold_nm
  fold_nm = f'{exp_nm}' if bool(exp_nm != '') else ''

  # Set up model
  print('Setting up model ...')
  model = LowLevelGPModel(data_stats).to(device)

  print('Created parameters:')
  total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  for param in model.parameters():
    print(type(param.data), param.shape)

  # Set up optimizers
  sgd_optimizer = pyro.optim.SGD({
    'lr': hparams['learning_rate'],
    'momentum': hparams['momentum'],
    'weight_decay': hparams['weight_decay'],
  })
  optimizer = sgd_optimizer

  create_model_dir()
  copy_model_script()
  print_and_log(f'hparams: {custom_hyperparams}')
  print_and_log(f'Total num. model parameters: {total_params}')
  print_and_log(f'Custom folder name: {exp_nm}')
  print_and_log(f'Dataset name: {dataset_nm}')

  train_model(model, optimizer, data)
  return


'''
  qsub
'''
def gen_qsubs(modelexp_nm = ''):
  # Generate qsub shell scripts and commands for easy parallelization
  print('Generating qsub scripts...')
  qsubs_dir = _config.QSUBS_DIR + NAME + '/'
  util.ensure_dir_exists(qsubs_dir)
  qsub_commands = []

  if modelexp_nm == '':
    modelexp_nm = 'modelexp_simple'

  print(f'Writing qsubs for {modelexp_nm}. OK?')
  input()

  exp_design = pd.read_csv(_config.DATA_DIR + f'{modelexp_nm}.csv')
  hyperparam_cols = [col for col in exp_design.columns if col != 'Name']

  # Parse df into dict
  hyperparam_combinations = dict()
  for idx, row in exp_design.iterrows():
    nm = row['Name']
    hps = '+'.join([f'{hp}:{row[hp]}' for hp in hyperparam_cols])
    hyperparam_combinations[nm] = hps

  # Generate qsubs
  num_scripts = 0
  for hyperparam_nm in hyperparam_combinations:
    hyperparam_setting = hyperparam_combinations[hyperparam_nm]

    command = f'python {NAME}.py {modelexp_nm} {hyperparam_nm} {hyperparam_setting}'
    script_id = NAME.split('_')[0]

    # Write shell scripts
    sh_fn = qsubs_dir + f'q_{modelexp_nm}_{hyperparam_nm}.sh'
    with open(sh_fn, 'w') as f:
      f.write('#!/bin/bash\n%s\n' % (command))
    num_scripts += 1

    # Write qsub commands
    qsub_commands.append(f'qsub -V -P regevlab -l h_rt=16:00:00,h_vmem=4G -l os=RedHat7 -wd {_config.SRC_DIR} {sh_fn} &')

  # Save commands
  commands_fn = qsubs_dir + '_commands.sh'
  with open(commands_fn, 'w') as f:
    f.write('\n'.join(qsub_commands))

  subprocess.check_output(f'chmod +x {commands_fn}', shell = True)
  print(f'Wrote {num_scripts} shell scripts to {qsubs_dir}')
  return


##
# Main
##
@util.time_dec
def main(argv = []):
  print(NAME)

  if argv[0] == 'test':
    run_inference()
  else:
    [modelexp_nm, hyperparam_nm, hyperparam_setting] = argv
    run_inference(
      batch_nm = modelexp_nm, 
      exp_nm = hyperparam_nm, 
      custom_hyperparams = hyperparam_setting,
    )

  return


if __name__ == '__main__':
  if len(sys.argv) >= 4:
    main(argv = sys.argv[1:])
  elif len(sys.argv) == 1:
    # print(f'Usage: python x.py <exp_nm>')
    # gen_qsubs()
    main(argv = ['test'])
  elif len(sys.argv) == 2:
    main(argv = ['test'])
