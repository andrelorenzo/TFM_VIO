#include "feature.hpp"

Feature::Feature(const int nFeatureId, const int nImageId)
{
    mnFeatureId = nFeatureId;
    mnRootImageId = nImageId;

    mbIsInited = false;
    mbIsMarginalized = false;
}