# Domino
Implementation of the [domino](https://en.wikipedia.org/wiki/Dominoes) game with support for the 6x6 and 9x9 versions.

# Dependencies
To able to run the domino game you will need to have [python](https://www.python.org/) installed.
To install the requirements run the following command:
```bash
pip install -r requirements.txt
```

# Vocabulary
- `Player`: An implementation of a domino player strategy
- `Rule`: An implementation of a domino game mode 
- `hands`: Implementations of special pieces handouts
- `behaviors`: Implementations of different player modifiers for special game situations

# Execution
A set of different modes to run domino plays. See the help command for listing them all.
```bash
python domino.py --help
```

## Info mode
All the game information that you need to know to run custom games is availble using the command:
```bash
python domino.py info
```
The output of this command show all the players, rules, hands and behaviors implemented.

## Play mode
To run and play the game you can use the following command:
```bash
python domino.py play
```
This command runs a default game between `Random` players.
To know how to run customs plays refer to the command help:
```bash
python domino.py play --help
```
## Match mode
This mode is meant to make a match between player and a list of oponents. See the command help for usage details.
```bash
python domino.py match --help
```

# Tournaments
There are some interesting touraments implemented that can be executed with differnt configurations.
To list the available tournaments use:
```bash
python tournaments.py --help
```
And then use the help of each tournament mode to know their usage like in the previous section.

# Train
To train the cooperAItive agent use:
```bash
python train.py config
```
Where config is the path to the training configuration file.
That configuration file should be a json with the following fields:
```json
{
    "batch_size": int,        //
    "handouts": int,          //
    "rollouts": int,          //
    "alpha": float,           //
    "max_number": int,        //
    "pieces_per_player": int, //
    "data_path": str,         //
    "save_path": str,         // Comments
    "lr": float,              //   for
    "tau_threshold": int,     //   each
    "epochs": int,            // property
    "simulate": bool,         //
    "load_checkpoint": bool,  //
    "load_model": bool,       //
    "sample": any,            //
    "tag": str,               //
    "num_process": int,       //
    "verbose": bool,          //
    "save_data": bool,        //
} 
```

# Development
You can add your own `player`, `rules`, `hands` and `behaviors` to the game.

## Players 
To add player create a file `module/players/strategies/my_custom_player.py` with a class that inherits from the `BasePlayer` defined in `module/palyers/player.py`.

Example:
```python
from ..player import BasePlayer

class MyCustomPlayer(BasePlayer):
    ...

    def filters(self, valids=None):
        return self.valid_moves() if valids is None else valids
```

To make your player available in the execution commands you need to register the player in the `PLAYERS` list that you can find in `module/players/strategies/__init__.py`

```python
...
from .my_custom_player import MyCustomPlayer

PLAYERS = [
    ...,
    MyCustomPlayer,
]
```

## Rules
Same as the previous section, create a file `module/rules/my_rule.py`, inherits from `BaseRule` located at `module/rules/rules.py` and register your rule in the `RULES` list that you can find in `module/rules/__init__.py`

## Hands
To add hand create a file `module/players/strategies/hands/my_hand.py`. 

Hands are functions that receive the game the `max_number` and the `pieces_per_player` of the game an return a list of 4 `PlayerView` objects

See the [hand_out](./module/players/hands/hand_out.py) file for reference.

You will need to register your `hand` in the `HAND` list of 
`module/players/strategies/hands/__init__.py`. 

## Behaviors
The behaviors are player implementation under the `module/players/behaviors/` folder and registered in the `BEHAVIORS` list of `module/players/behaviors/__init__.py`

See the `player` section above for reference. 