#pragma once
#include <vector>

#include <Eigen/Core>

#include <opencv2/core/core.hpp>

#include "config.hpp"


class Feature
{
public:

    Feature(const int nFeatureId, const int nImageId);

    inline void Inited() {mbIsInited = true;}

    inline void Marginalized() {mbIsMarginalized = true;}

    inline void SetPosition(const vec3& position) {mPosition = position;}

    inline void SetFejPosition(const vec3& position) {mFejPosition = position;}

    inline const int FeatureId() const {return mnFeatureId;}

    inline const int RootImageId() const {return mnRootImageId;}

    inline const bool IsInited() const {return mbIsInited;}

    inline const bool IsMarginalized() const {return mbIsMarginalized;}

    inline vec3& Position() {return mPosition;}

    inline vec3& FejPosition() {return mFejPosition;}

    inline void reset(const int nImageId)
    {
        mnRootImageId = nImageId;
        mbIsInited = false;
        mbIsMarginalized = false;
    }

    inline void clear()
    {
        mnRootImageId = -1;
        mbIsInited = false;
        mbIsMarginalized = false;
    }

private:

    int mnFeatureId;   // start from 0
    int mnRootImageId; // start from 0

    bool mbIsInited;
    bool mbIsMarginalized;

    vec3 mPosition;
    vec3 mFejPosition;
};
