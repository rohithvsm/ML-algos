import java.util.HashMap;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Collections;

public class AStarSearchImpl implements AStarSearch {
	
	@Override	
	public SearchResult search(String initConfig, int modeFlag) {

        int initStateHeuristic = getHeuristicCost(initConfig, modeFlag);
        //System.out.println(initStateHeuristic);
        State initState = new State(initConfig, 0, initStateHeuristic, "");
        int numPoppedStates = 0;
        //System.out.println(checkGoal("111132323132231222312133"));

        PriorityQueue<State> open   = new PriorityQueue<State>(State.comparator);
        //PriorityQueue<State> closed = new PriorityQueue<State>(State.comparator);
        HashMap<String,State> openHash   = new HashMap<String,State>();
        HashMap<String,State> closedHash = new HashMap<String,State>();
        open.add(initState);
        openHash.put(initState.config, initState);
        //System.out.println("Test move");
        //System.out.println(move("113132223132231222313113", 'C'));
        //System.out.println(move(move(initConfig, 'A'), 'C'));
        while(open.size() != 0) {
            State n = open.poll();
            openHash.remove(n.config);
            //closed.add(n);
            closedHash.put(n.config, n);
            //System.out.println(n.opSequence);
            //System.out.println(n.config);
            numPoppedStates++;
            if(checkGoal(n.config)) {
                //System.out.println(n.config);
                //System.out.println(n.opSequence);
                //System.out.println(numPoppedStates);
                //System.out.println(closed.size());
                return new SearchResult(n.config, n.opSequence, numPoppedStates);
            }
            // Expand n
            char moves[] = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'};
            for(int i = 0; i < moves.length; i++) {
                String rotatedConfig = move(n.config, moves[i]);
                //if(moves[i] == 'C')
                    //System.out.println("C: " + rotatedConfig);
                //System.out.println(Character.toString(moves[i]));
                State nDash = new State(rotatedConfig, n.realCost + 1, getHeuristicCost(rotatedConfig, modeFlag), n.opSequence+Character.toString(moves[i]));
                //if(!(open.contains(nDash) || closed.contains(nDash))) {
                //if(!open_closed.containsKey(nDash.config)) {
                if(!(openHash.containsKey(nDash.config) || closedHash.containsKey(nDash.config))) {
                    open.add(nDash);
                    //open_closed.put(nDash.config, nDash);
                    openHash.put(nDash.config, nDash);
                }
                else {
                    //if(open.contains(nDash)) {
                    if(openHash.containsKey(nDash.config)) {
                        /*
                        Iterator<State> it = open.iterator();
                        while(it.hasNext()) {
                            State check = it.next();
                            if(nDash == check) {
                                if(nDash.realCost < check.realCost) {
                                    open.remove(check);
                                    open.add(nDash);
                                }
                            }
                        }
                        */
                        if(nDash.realCost < openHash.get(nDash.config).realCost) {
                            openHash.get(nDash.config);
                            open.remove(nDash);
                            openHash.put(nDash.config, nDash);
                            open.add(nDash);
                        }
                    }
                    //else if(closed.contains(nDash)) {
                    else if(closedHash.containsKey(nDash.config)) {
                        /*
                        Iterator<State> it = closed.iterator();
                        while(it.hasNext()) {
                            State check = it.next();
                            if(nDash == check) {
                                if(nDash.realCost < check.realCost) {
                                    closed.remove(check);
                                    open.add(nDash);
                                }
                            }
                        }
                        */
                        if(nDash.realCost < closedHash.get(nDash.config).realCost) {
                            closedHash.remove(nDash.config);
                            //closed.remove(nDash);
                            openHash.put(nDash.config, nDash);
                            open.add(nDash);
                        }
                    }
                }
            }
            //System.out.println(open.size());
        }
		return null;
	}

	@Override
	public boolean checkGoal(String config) {
		char centralSquare[] = {config.charAt(6) , config.charAt(7) , config.charAt(8) , config.charAt(11)
                               ,config.charAt(12), config.charAt(15), config.charAt(16), config.charAt(17)};
        char element = centralSquare[0];
        for(int i = 1; i < 8; i++) {
            if(centralSquare[i] != element)
                return false;
            element = centralSquare[i];
        }
		return true;
	}

	@Override
	public String move(String config, char op) {	

        char[] rotatedConfig = config.toCharArray();
        int[] aIndices = {0, 2, 6, 11, 15, 20, 22};
        int[] bIndices = {1, 3, 8, 12, 17, 21, 23};
        int[] hIndices = {4, 5, 6, 7, 8, 9, 10};
        int[] gIndices = {13, 14, 15, 16, 17, 18, 19};
        int[] indices = {};
        //System.out.println(rotatedConfig);
        if(op == 'A')
            indices = aIndices;
        else if(op == 'B')
            indices = bIndices;
        else if(op == 'H')
            indices = hIndices;
        else if(op == 'G')
            indices = gIndices;
        else if(op == 'F')
            //System.out.println(Arrays.toString(reverseArray(aIndices)));
            indices = reverseArray(aIndices);
        else if(op == 'E')
            indices = reverseArray(bIndices);
        else if(op == 'C')
            indices = reverseArray(hIndices);
        else if(op == 'D')
            indices = reverseArray(gIndices);

        char head = rotatedConfig[indices[0]];
        for(int i = 0; i < indices.length - 1; i++) {
            rotatedConfig[indices[i]] = rotatedConfig[indices[i+1]];
        }
        rotatedConfig[indices[indices.length - 1]] = head;
        //System.out.println(rotatedConfig);
        return new String(rotatedConfig);
	}

	@Override
	public int getHeuristicCost(String config, int modeFlag) {		
		
        if(modeFlag == 2)
            return 0;

        int heuristic = 0;
        char centralSquare[] = {config.charAt(6) , config.charAt(7) , config.charAt(8) , config.charAt(11)
                               ,config.charAt(12), config.charAt(15), config.charAt(16), config.charAt(17)};
        int count_1 = 0;
        int count_2 = 0;
        int count_3 = 0;
        for(int i = 0; i < centralSquare.length; i++) {
            if(centralSquare[i] == '1')
                count_1 += 1;
            else if(centralSquare[i] == '2')
                count_2 += 1;
            else if(centralSquare[i] == '3')
                count_3 += 1;
        }

        if(modeFlag == 1) {
            //System.out.println(centralSquare);

            /*
            System.out.println(count_1);
            System.out.println(count_2);
            System.out.println(count_3);
            */
            heuristic =  8 - (Math.max(Math.max(count_1, count_2), count_3));
        }

        else if(modeFlag == 3) {
            int[] counts = {count_1, count_2, count_3};
            Arrays.sort(counts);
            heuristic = counts[0] + counts[1];
        }

        else if(modeFlag == 4) {
            int[] counts = {count_1, count_2, count_3};
            Arrays.sort(counts);
            heuristic = 8 - (counts[2] - counts[1] - counts[0]);
        }

        return heuristic;
	}
	
    private int[] reverseArray(int[] array) {
        for(int i =0; i < array.length/2; i++) {
            int temp = array[i];
            array[i] = array[array.length - 1 - i];
            array[array.length - 1 - i] = temp;
        }
    return array;
    }

}
