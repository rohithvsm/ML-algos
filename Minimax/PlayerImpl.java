import java.util.*;

public class PlayerImpl implements Player {
	// Identifies the player
	private int name = 0;
	int n = 0;

	// Constructor
	public PlayerImpl(int name, int n) {
		this.name = 0;
		this.n = n;
	}

	// Function to find possible successors
	@Override
	public ArrayList<Integer> generateSuccessors(int lastMove, int[] crossedList) {

        ArrayList<Integer> nextValidMoves = new ArrayList<>();

        // first move
        if(lastMove == -1) {
            float res = (float) n / 2;
            for(int i = 2; i < res; i = i + 2)
                nextValidMoves.add(i);

            return nextValidMoves;
        }

        for(int i = 1; i <= n; i++)
            if(crossedList[i] == 0)
                if(i % lastMove == 0)
                    nextValidMoves.add(i);
                else if(lastMove % i == 0)
                    nextValidMoves.add(i);

		return nextValidMoves;
	}

	// The max value function
	@Override
	public int max_value(GameState s) {

        if(s.leaf)
            return -1;

        // children
        ArrayList<Integer> nextValidMoves = new ArrayList<>();
        nextValidMoves = generateSuccessors(s.lastMove, s.crossedList);

        int alpha = Integer.MIN_VALUE;
        for(int i = 0; i < nextValidMoves.size(); i++) {
            int nextValidMove = nextValidMoves.get(i);
            GameState newGameState = new GameState(s.crossedList, nextValidMove);
            newGameState.crossedList[nextValidMove] = 1;  // mark it crossed

            int new_val = min_value(newGameState);
            if(new_val >= alpha) {
                alpha = new_val;
                s.bestMove = nextValidMove;
            }
        }

        if(alpha == Integer.MIN_VALUE) {
            s.leaf = true;
            s.bestMove = -1;
            alpha = -1;
        }

		return alpha;
	}

	// The min value function
	@Override
	public int min_value(GameState s) {

        if(s.leaf)
            return 1;

        // children
        ArrayList<Integer> nextValidMoves = new ArrayList<>();
        nextValidMoves = generateSuccessors(s.lastMove, s.crossedList);

        int beta = Integer.MAX_VALUE;
        for(int i = 0; i < nextValidMoves.size(); i++) {
            int nextValidMove = nextValidMoves.get(i);
            GameState newGameState = new GameState(s.crossedList, nextValidMove);
            newGameState.crossedList[nextValidMove] = 1;  // mark it crossed

            int new_val = max_value(newGameState);
            if(new_val <= beta) {
                beta = new_val;
                s.bestMove = nextValidMove;
            }
        }

        if(beta == Integer.MAX_VALUE) {
            s.leaf = true;
            s.bestMove = -1;
            beta = 1;
        }

		return beta;
	}

	// Function to find the next best move
	@Override
	public int move(int lastMove, int[] crossedList) {

        GameState newState = new GameState(crossedList, lastMove);
        max_value(newState);

		return newState.bestMove;
	}
}
