function [ X ] = uv2xyz_1( circleParaXYR1, circleParaXYR, leftIntrisic, leftTranslation, leftRotation, rightIntrisic, rightTranslation, rightRotation)
mLeftRT = [];
mLeftM  = [];
mLeftRT = [leftRotation,leftTranslation'];
mLeftM = leftIntrisic*mLeftRT;
mRightRT = [];
mRightM = [];
mRightRT = [rightRotation,rightTranslation'];
mRightM  = rightIntrisic*mRightRT;

A = [];
A(1,1) = circleParaXYR1(1,2) * mLeftM(3,1) - mLeftM(1,1);
A(1,2) = circleParaXYR1(1,2) * mLeftM(3,2) - mLeftM(1,2);
A(1,3) = circleParaXYR1(1,2) * mLeftM(3,3) - mLeftM(1,3);

A(2,1) = circleParaXYR1(1,1) * mLeftM(3,1) - mLeftM(2,1);
A(2,2) = circleParaXYR1(1,1) * mLeftM(3,2) - mLeftM(2,2);
A(2,3) = circleParaXYR1(1,1) * mLeftM(3,3) - mLeftM(2,3);

A(3,1) = circleParaXYR(1,2) * mRightM(3,1) - mRightM(1,1);
A(3,2) = circleParaXYR(1,2) * mRightM(3,2) - mRightM(1,2);
A(3,3) = circleParaXYR(1,2) * mRightM(3,3) - mRightM(1,3);

A(4,1) = circleParaXYR(1,1) * mRightM(3,1) - mRightM(2,1);
A(4,2) = circleParaXYR(1,1) * mRightM(3,2) - mRightM(2,2);
A(4,3) = circleParaXYR(1,1) * mRightM(3,3) - mRightM(2,3);

B = [];
B(1,1) = mLeftM(1,4) - circleParaXYR1(1,1) * mLeftM(3,4);
B(1,2) = mLeftM(2,4) - circleParaXYR1(1,2)* mLeftM(3,4);
B(1,3) = mRightM(1,4) - circleParaXYR(1,1) * mRightM(3,4);
B(1,4) = mRightM(2,4) - circleParaXYR(1,2)* mRightM(3,4);

X = [];
X = A\B';
end

