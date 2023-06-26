//import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
//import java.util.stream.IntStream;
import java.util.Comparator;

public class SimpleTesting{
    //******优化排列组合*******//
    /**
     * 替换阶乘的另一种方式(从n开始递减相乘，乘m个数。即 n*(n-1)...(n-m+) 结束)
     */
    private static long factorialSec(int m, int n) {
        long sum = 1;
 
        while(m > 0 && n > 0) {
            sum = sum * n--;
            m--;
        }
        return sum;
    }
 
    /**
     * 排列
     */
    public static long arrangementSec(int m, int n) {
        return m <= n ? factorialSec(m, n) : 0;
    }
 
    /**
     * 组合
     */
    public static long combinationSec(int m, int n) {
        if( m > n )
            return 0;
        if( m > n/2 )
        {
            m = n - m;
        }
        return factorialSec(m, n)/factorialSec(m, m);
    }

    public static List<List<Integer>> getRemoveLists(List<String> methodRemoveList, int maxSampleNumber){
        int max_val = methodRemoveList.size();
        //System.out.println(max_val);
        
        Random randInt = new Random();
        List<List<Integer>> samples = new ArrayList<List<Integer>>();
        for (int i = 0; i < max_val; i++){

            // long sampleNumber = combinationSec(i+1,max_val);
            // if (sampleNumber > 20){
            //     sampleNumber = 20;
            // }
            for (int j = 0; j < 20; j++){ 
                List<Integer> removeList = new ArrayList<Integer>();
                while (removeList.size() < i){
                    int whileNum = 0;
                    int randindex = randInt.nextInt(max_val);
                    if (!removeList.contains(randindex)){
                        removeList.add(randindex);
                    }
                    if (whileNum > 20){
                        break;
                    }
                }
                removeList.sort(Comparator.naturalOrder());
                if (!samples.contains(removeList))
                    samples.add(removeList);
            }
        }
        
        //System.out.println(samples);
        //System.out.println(samples.size());


        //从所有结果中采样至多20个删除方案
        if(samples.size() < maxSampleNumber){
            maxSampleNumber = samples.size();
        }
        
        List<Integer> sampleList = new ArrayList<Integer>();
        while(sampleList.size() < maxSampleNumber){
            int whileNum = 0;
            int randIndex = randInt.nextInt(samples.size());
            if(!sampleList.contains(randIndex)){
                sampleList.add(randIndex);
            }
            sampleList.sort(Comparator.naturalOrder());
            whileNum+=1;
            if (whileNum > 20){
                break;
            }
        }
        //System.out.println(sampleList);
        List<List<Integer>> finalSamples = new ArrayList<List<Integer>>();
        for(int i = 0; i < sampleList.size(); i++){
            finalSamples.add(samples.get(sampleList.get(i)));
        }
        //System.out.println(finalSamples);
        //System.out.println(finalSamples.size());
        return finalSamples;
    }

    //******优化结束******//
    public static void main(String[] args) { 
        List<String> methodRemoveList = new ArrayList<>();
        methodRemoveList.add("dealerPhase");
        methodRemoveList.add("main");
        methodRemoveList.add("initialDeal");
        methodRemoveList.add("inputFromPlayer");
        methodRemoveList.add("displayGameResults");
        methodRemoveList.add("displayGameState");
        methodRemoveList.add("play");
        methodRemoveList.add("displayFinalGameState");
        methodRemoveList.add("displayHandFormatted");

        List<List<Integer>> removeLists = getRemoveLists(methodRemoveList, 20);
        System.out.println(removeLists);
        System.out.println(removeLists.size());
    }
}
