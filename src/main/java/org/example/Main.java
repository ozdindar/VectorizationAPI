package org.example;


import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorSpecies;

import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;


public class Main {

    static final VectorSpecies<Integer> SPECIES = IntVector.SPECIES_PREFERRED;
    //static final VectorSpecies<Float> SPECIES_F = FloatVector.SPECIES_256;

    public static int[] addTwoVectorArrays(int[] arr1, int[] arr2) {
        var v1 = IntVector.fromArray(SPECIES, arr1, 0);
        var v2 = IntVector.fromArray(SPECIES, arr2, 0);
        var result = v1.add(v2);
        DoubleVector d;

        return result.toArray();
    }


    public static int[] addTwoVectorsWithMasks(int[] arr1, int[] arr2) {
        int[] finalResult = new int[arr1.length];
        int i = 0;
        for (; i < SPECIES.loopBound(arr1.length); i += SPECIES.length()) {
            var mask = SPECIES.indexInRange(i, arr1.length);
            var v1 = IntVector.fromArray(SPECIES, arr1, i, mask);
            var v2 = IntVector.fromArray(SPECIES, arr2, i, mask);
            var result = v1.add(v2, mask);
            result.intoArray(finalResult, i, mask);
        }

        // tail cleanup loop
        for (; i < arr1.length; i++) {
            finalResult[i] = arr1[i] + arr2[i];
        }
        return finalResult;
    }

/*
    static void vectorComputation(float[] a, float[] b, float[] c) {

        for (int i = 0; i < a.length; i += SPECIES.length()) {
            var m = SPECIES.indexInRange(i, a.length);
            // FloatVector va, vb, vc;
            var va = FloatVector.fromArray(SPECIES_F, a, i, m);
            var vb = FloatVector.fromArray(SPECIES_F, b, i, m);
            var vc = va.mul(va).
                    add(vb.mul(vb)).
                    neg();
            vc.intoArray(c, i, m);
        }
    }
*/

    public static void main(String[] args) {
        int[]  arr1 = IntStream.generate(() -> new Random().nextInt(100)).limit(100).toArray();
        int[]  arr2 = IntStream.generate(() -> new Random().nextInt(100)).limit(100).toArray();

        int[] arr3 = addTwoVectorsWithMasks(arr1,arr2);

        System.out.println(Arrays.toString(arr1));
        System.out.println(Arrays.toString(arr2));
        System.out.println(Arrays.toString(arr3));

/*        float[] a = {1.0f, 2.0f, 3.0f, 4.0f};
        float[] b = {1.0f, 2.0f, 3.0f, 4.0f};
        float[] c = {1.0f, 2.0f, 3.0f, 4.0f};

        vectorComputation(a, b, c);*/
    }
}