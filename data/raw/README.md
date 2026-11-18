# Raw data

Put the six supplied CSV files here before running the pipeline:

- `ball_by_ball.csv`
- `matches.csv`
- `competitions.csv`
- `teams.csv`
- `playing_eleven.csv`
- `competition_team_player_mapping.csv`

They are listed in `.gitignore`, so they are never committed. To read them from somewhere else, set `SUPERR_DATA`, for example `SUPERR_DATA=/path/to/csvs/ python pipeline/pressure_v2.py`.
