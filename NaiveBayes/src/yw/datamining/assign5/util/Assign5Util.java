package yw.datamining.assign5.util;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

public class Assign5Util {

	public static Map<Integer, Integer> CNUM = new HashMap<Integer, Integer>();
	
	public static List<List<Integer>> validDataSet(List<List<Double>> examples, List<Integer> labels, Integer Num){
		
		int size = examples.size();
		Map<Integer, List<Integer>> tMap = new HashMap<>();
		for(int i = 0; i < size; i++){
			Integer label = labels.get(i);
			if(!tMap.containsKey(label)){
				tMap.put(label, new ArrayList<Integer>());
			}
			tMap.get(label).add(i);
		}
		
		Random random = new Random();
		
		List<List<Integer>> result = new ArrayList<>();
		
		for(int i = 0; i < Num; i++){
			
			List<Integer> tList = new ArrayList<>();
			
			for(Map.Entry<Integer, List<Integer>> entry : tMap.entrySet()){
				random.nextInt(entry.getValue().size());
				int temp = entry.getValue().size()/Num;
				if(i < Num - 1){
					tList.addAll(entry.getValue().subList(i*temp, (i+1)*temp));
				}else{
					tList.addAll(entry.getValue().subList(i*temp, entry.getValue().size()));
				}
			}
			
			result.add(tList);
		}
		return result;
	}
	
	/**
	 * 朴素贝叶斯算法中计算先验概率
	 * @param labels
	 * @return
	 */
	public static Map<Integer, Double> Pc(List<Integer> labels){
		
		int num = labels.size();
		
		Map<Integer, Double> result = new HashMap<>();
		for(Integer label : labels){
			if(!result.containsKey(label)){
				CNUM.put(label, 1);
				result.put(label, 1.0);
			}else{
				CNUM.put(label, CNUM.get(label) + 1);
				result.put(label, result.get(label) + 1.0);
			}
		}
		result.keySet().forEach(key->result.put(key, (result.get(key)+1)/(num + result.size())));
		return result;
	}
	
	/**
	 * 朴素贝叶斯算法中计算条件概率
	 * @param examples
	 * @param labels
	 * @param featureType
	 * @return
	 */
	public static Map<Integer, Map<Integer, Map<Double, Double>>> pxic(
			List<List<Double>> examples, List<Integer> labels, List<Integer> featureType){
		
		int rowNum = examples.size();
		int colNum = examples.get(0).size();
		
		Set<Integer> labelsSet = new HashSet<>(labels);
		
		Map<Integer, Map<Integer, Map<Double, Double>>> result = new HashMap<>();
		
		for(int i = 0; i < colNum; i++){
			/**
			 * 离散属性
			 */
			if(featureType.get(i) == 1){
				/**
				 * 先遍历一遍，确定属性i都有哪些取值
				 */
				Map<Integer, Map<Double, Double>> tMap = new HashMap<>();
				for(Integer label : labelsSet){
					tMap.put(label, new HashMap<Double, Double>());
				}
				
				for(int j = 0; j < rowNum; j++){
					
					double data = examples.get(j).get(i);
					int label = labels.get(j);
					for(Integer tInt : labelsSet){
						if(!tMap.get(tInt).containsKey(data)){
							tMap.get(tInt).put(data, 0.0);
						}
						if(tInt == label){
							tMap.get(tInt).put(data, tMap.get(tInt).get(data) + 1.0);
						}
					}
				}
				
				for(Integer tInt : tMap.keySet()){
					Map<Double, Double> ttMap = tMap.get(tInt);
					for(Double tDouble : ttMap.keySet()){
						double temp = ttMap.get(tDouble);
						tMap.get(tInt).put(tDouble, (temp + 1.0)/(CNUM.get(tInt) + tMap.get(tInt).size()));
					}
				}
				
				result.put(i, tMap);
				
			}else{
				Map<Integer, Map<Double, Double>> tMap = new HashMap<>();
				for(Integer label : labelsSet){
					tMap.put(label, new HashMap<Double, Double>());
				}
				
				Map<Integer, Double> meanMap = new HashMap<>();
				for(Integer tLabel : labelsSet){
					meanMap.put(tLabel, 0.0);
				}
				for(int j = 0; j < rowNum; j++){
					int label = labels.get(j);
					meanMap.put(label, meanMap.get(label) + examples.get(j).get(i));
				}
				meanMap.keySet().forEach(key->meanMap.put(key, meanMap.get(key)/CNUM.get(key)));
				
				Map<Integer, Double> varianceMap = new HashMap<>();
				for(Integer tLabel : labelsSet){
					varianceMap.put(tLabel, 0.0);
				}
				for(int j = 0; j < rowNum; j++){
					int label = labels.get(j);
					varianceMap.put(label, varianceMap.get(label) 
							+ Math.pow(examples.get(j).get(i) - meanMap.get(label), 2));
				}
				varianceMap.keySet().forEach(key->varianceMap.put(key, varianceMap.get(key)/CNUM.get(key)));
				
				for(int j = 0; j < rowNum; j++){
					double data = examples.get(j).get(i);
					
					for(Integer label : labelsSet){
						double temp = Math.exp(-(Math.pow(data-meanMap.get(label), 2)/(2*varianceMap.get(label))))
								/ (Math.pow(2*Math.PI, 0.5)*Math.pow(varianceMap.get(label), 0.5));
						tMap.get(label).put(data, temp);
					}
					
				}
				result.put(i, tMap);
			}
		}
		return result;
	}
	
	public static List<Integer> classifier(List<List<Double>> examplesTrain, 
			List<List<Double>> examplesTest, List<Integer> labels, List<Integer> featureType){
		Map<Integer, Double> PC = Assign5Util.Pc(labels);
		Map<Integer, Map<Integer, Map<Double, Double>>> PXIC = Assign5Util.pxic(examplesTrain, labels, featureType);
		
		List<Integer> result = new ArrayList<>();
		
		int rowNum = examplesTest.size();
		int colNum = examplesTest.get(0).size();
		
		for(int i = 0; i < rowNum; i++){
			
			Map<Integer, Double> ttMap = new HashMap<>();
			
			for(int j = 0; j < colNum; j++){
				double data = examplesTest.get(i).get(j);
				Map<Integer, Map<Double, Double>> temp = PXIC.get(j);
				for(Map.Entry<Integer, Map<Double, Double>> entry : 
					temp.entrySet()){
					Integer label = entry.getKey();
					Map<Double, Double> t = entry.getValue();
					
					Double tPxic = null;
					
					if(!t.containsKey(data)){
						tPxic = Double.valueOf(1.0/(CNUM.get(label) + PXIC.get(j).get(label).size()));
					}else{
						tPxic = t.get(data);
					}
					
					if(!ttMap.containsKey(label)){
						ttMap.put(label, tPxic);
					}else{
						ttMap.put(label, ttMap.get(label) * tPxic);
					}
				}
			}
			Double tResult = 0.0;
			Integer tKey = 0;
			for(Map.Entry<Integer, Double> entry : ttMap.entrySet()){
				Integer ttKey = entry.getKey();
				Double ttValue = entry.getValue();
				Double midValue = Double.valueOf(PC.get(ttKey)*ttValue);
				ttMap.put(ttKey, midValue);
				if(midValue.compareTo(tResult) > 0){
					tResult = midValue;
					tKey = ttKey;
				}
			}
			result.add(tKey);
		}
		
		return result;
	}
	
	public static Double errorValue(List<Integer> realResult, List<Integer> calculateResult){
		int num = 0;
		for(int i = 0; i < realResult.size(); i++){
			if(realResult.get(i) != calculateResult.get(i)) num++;
		}
		return Double.valueOf(num)/realResult.size();
	}
	
	public static List<Integer> getSubDataSet(List<List<Double>> examples, List<Double> weight, int NUM ){
		int size = weight.size();
		List<Integer> indexs = new ArrayList<>();
		Map<Integer, Double> tMap = new HashMap<>();
		tMap.put(-1, 0.0);
		for(int i = 0; i< size; i++){
			tMap.put(i, tMap.get(i-1) + weight.get(i));
		}
		Random random = new Random();
		do{
			Double randomDouble = random.nextDouble()*tMap.get(size-1);
			for(int i = 0; i < size; i++){
				if(randomDouble.compareTo(tMap.get(i-1)) > 0
						&& randomDouble.compareTo(tMap.get(i)) < 0){
					indexs.add(i);
				}
			}
		}while(indexs.size() < NUM);
		
		return indexs;
	}
}
