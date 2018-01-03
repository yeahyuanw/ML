package yw.datamining.assign5.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

/**
 * 从文件中读数据集
 * @author ywang
 * @version 2016/11/23
 */
public class ReadFile {

	/**
	 * representing discrete feature and 0 representing numerical feature
	 */
	private List<Integer> featureType;
	
	/**
	 * the label of the corresponding example
	 */
	private List<Integer> labels;
	
	/**
	 * the corresponding example
	 */
	private List<List<Double>> examples;
	
	public ReadFile(String filePath){
		
		this.featureType = new ArrayList<>();
		this.labels = new ArrayList<>();
		this.examples = new ArrayList<>();
		
		File file = new File(filePath);
		if(file.exists()){
			try(BufferedReader bufferedReader
					= new BufferedReader(new FileReader(file))){
				String flag = bufferedReader.readLine();
				if(flag != null && !"".equals(flag)){
					
					for(String str : flag.split(",")){
						this.featureType.add(Integer.valueOf(str));
					}
				}
				flag = bufferedReader.readLine();
				while(flag != null && !"".equals(flag)){
					
					String[] temp = flag.split(",");
					List<Double> tList = new ArrayList<>();
					
					for(int i = 0; i < temp.length -1; i++){
						tList.add(Double.valueOf(temp[i]));
					}
					
					this.examples.add(tList);
					this.labels.add(Double.valueOf(temp[temp.length - 1]).intValue());
					flag = bufferedReader.readLine();
				}
			}catch (Exception e) {
				e.printStackTrace();
			}
		}
	}

	public List<Integer> getFeatureType() {
		return featureType;
	}

	public List<Integer> getLabels() {
		return labels;
	}

	public List<List<Double>> getExamples() {
		return examples;
	}

}
