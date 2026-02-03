// -*- C++ -*-
//
// Copyright (C) 2020-2025  MACESW developers
//
// This file is part of MACESW, Muonium-to-Antimuonium Conversion Experiment
// offline software.
//
// MACESW is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version.
//
// MACESW is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
// A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with
// MACESW. If not, see <https://www.gnu.org/licenses/>.

#include "MACE/Data/SimHit.h++"
#include "MACE/Detector/Description/ECAL.h++"
#include "MACE/PhaseI/Detector/Description/UsePhaseIDefault.h++"
#include "MACE/PhaseI/ReconECAL/ReconECAL.h++"
#include "MACE/Reconstruction/ECALClustering/Clusterer.h++"

#include "Mustard/CLI/BasicCLI.h++"
#include "Mustard/Data/Output.h++"
#include "Mustard/Data/Processor.h++"
#include "Mustard/Data/Tuple.h++"
#include "Mustard/Detector/Description/DescriptionIO.h++"
#include "Mustard/Env/MPIEnv.h++"
#include "Mustard/Parallel/ProcessSpecificPath.h++"
#include "Mustard/Utility/LiteralUnit.h++"
#include "Mustard/Utility/MathConstant.h++"
#include "Mustard/Utility/PhysicalConstant.h++"
#include "Mustard/Utility/VectorArithmeticOperator.h++"

#include "CLHEP/Vector/ThreeVector.h"

#include "ROOT/RDataFrame.hxx"
#include "TFile.h"
#include "TH1.h"
#include "TH3.h"
#include "TRandom.h"
#include "TTree.h"
#include "TVector3.h"

#include "muc/algorithm"

#include "fmt/format.h"

#include <algorithm>
#include <functional>
#include <ranges>
#include <unordered_map>
#include <unordered_set>

namespace MACE::PhaseI::ReconECAL {

ReconECAL::ReconECAL() :
    Subprogram{"ReconECAL", "Electromagnetic calorimeter (ECAL) event reconstruction in PhaseI."} {}

using namespace Mustard::LiteralUnit;
using namespace Mustard::MathConstant;
using namespace Mustard::PhysicalConstant;
using namespace std::literals;

auto ReconECAL::Main(int argc, char* argv[]) const -> int {
    Mustard::CLI::BasicCLI<> cli;
    cli->add_argument("input").help("Input file path(s).").nargs(argparse::nargs_pattern::at_least_one);
    cli->add_argument("-t", "--input-tree").help("Input tree name.").default_value("G4Run0/ECALSimHit"s).required().nargs(1);
    cli->add_argument("-o", "--output").help("Output file path.").required().nargs(1);
    cli->add_argument("-m", "--output-mode").help("Output file creation mode.").default_value("RECREATE"s).required().nargs(1);
    cli->add_argument("-d", "--description").help("Description YAML file path.").nargs(1);
    cli->add_argument("-cali", "--recon-calibration").help("Reconstruction of calibration events.").flag();
    cli->add_argument("-two", "--recon-two-body").help("Reconstruction of two body decay events.").flag();
    cli->add_argument("-inv", "--recon-invisible").help("Reconstruction of invisible decay events.").flag();
    Mustard::Env::MPIEnv env{argc, argv, cli};

    const auto reconstructCalibration{cli["--recon-calibration"] == true};
    const auto reconstructTwoBody{cli["--recon-two-body"] == true};
    const auto reconstructInvisible{cli["--recon-invisible"] == true};
    std::vector<bool> reconstructionFlags{reconstructCalibration, reconstructTwoBody, reconstructInvisible};

    if (std::ranges::count(reconstructionFlags, true) != 1) {
        Mustard::PrintError("One and only one reconstruction mode must be enabled.");
    }

    if (const auto descriptionPath{cli->present("--description")}) {
        Mustard::Detector::Description::DescriptionIO::Import<MACE::Detector::Description::ECAL>(*descriptionPath);
    } else {
        Detector::Description::UsePhaseIDefault();
    }

    const auto& ecal{MACE::Detector::Description::ECAL::Instance()};
    const auto& moduleList{ecal.Array().moduleList};

    TFile outputFile{Mustard::Parallel::ProcessSpecificPath(cli->get("--output").c_str()).generic_string().c_str(), cli->get("--output-mode").c_str()};

    if (reconstructCalibration) {
        using ECALEnergy = Mustard::Data::TupleModel<
            Mustard::Data::Value<double, "Edep", "Energy deposition in total">,
            Mustard::Data::Value<int, "PE", "Photoelectron counts in total">,
            Mustard::Data::Value<muc::array3f, "Centroid", "Centroid of the first cluster">,
            Mustard::Data::Value<double, "cosTheta", "Angle between truth and reconstructed tracks">>;
        Mustard::Data::Output<ECALEnergy> reconEnergy{"G4Run0/ReconECAL"};

        Mustard::Data::Processor processor;
        processor.Process<Data::ECALSimHit>(
            ROOT::RDataFrame{cli->get("--input-tree"), cli->get<std::vector<std::string>>("input")}, int{}, "EvtID",
            [&](bool byPass, auto&& event) {
                if (byPass) {
                    return;
                }
                muc::timsort(event,
                             [](auto&& hit1, auto&& hit2) {
                                 return Get<"Edep">(*hit1) > Get<"Edep">(*hit2);
                             });

                std::unordered_map<int, std::shared_ptr<Mustard::Data::Tuple<Data::ECALSimHit>>> hitDict;
                std::vector<int> potentialSeedModule;
                muc::array3f truthHitMomentum{};

                for (auto&& hit : event) {
                    hitDict.try_emplace(Get<"ModID">(*hit), hit);
                    potentialSeedModule.emplace_back(Get<"ModID">(*hit));

                    if (Get<"TrkID">(*hit) == 1 and Get<"HitID">(*hit) == 0) {
                        truthHitMomentum = Get<"p0">(*hit);
                    }
                }

                CLHEP::Hep3Vector truthHitVector{
                    truthHitMomentum.at(0),
                    truthHitMomentum.at(1),
                    truthHitMomentum.at(2)};

                std::unordered_set<int> firstCluster;
                CLHEP::Hep3Vector firstClusterCentroid{};
                auto firstSeedModule = potentialSeedModule.begin();

                const auto clustering = [&](std::unordered_set<int>& set,
                                            CLHEP::Hep3Vector& c,
                                            std::vector<int>::iterator seedIt) {
                    const auto addClusterLayers = [&](int module) {
                        set.insert(module);
                        for (auto&& neighbor : moduleList.at(module).neighborModuleID) {
                            set.insert(neighbor);
                            for (auto&& secondNeighbor : moduleList.at(neighbor).neighborModuleID) {
                                set.insert(secondNeighbor);
                            }
                        }
                    };

                    addClusterLayers(*seedIt);

                    double totalEnergy{};
                    int totalPE{};
                    CLHEP::Hep3Vector weightedCentroid{};

                    for (const auto& module : set) {
                        auto hitIt = hitDict.find(module);

                        if (hitIt == hitDict.end() or Get<"Edep">(*hitIt->second) < 50_keV) {
                            continue;
                        }
                        auto energy = Get<"Edep">(*hitIt->second);
                        auto pe = Get<"nOptPho">(*hitIt->second);
                        if (pe > 3) {
                            weightedCentroid += energy * moduleList.at(module).centroid;
                            totalEnergy += energy;
                            totalPE += pe;
                        }
                    }
                    c = weightedCentroid / totalEnergy;

                    return std::make_pair(totalEnergy, totalPE);
                };

                auto firstClusterSignal = clustering(firstCluster, firstClusterCentroid, firstSeedModule);

                Mustard::Data::Tuple<ECALEnergy> energyTuple;
                Get<"Edep">(energyTuple) = firstClusterSignal.first;
                Get<"PE">(energyTuple) = firstClusterSignal.second;
                Get<"Centroid">(energyTuple) = firstClusterCentroid;
                Get<"cosTheta">(energyTuple) = firstClusterCentroid.cosTheta(truthHitVector);

                reconEnergy.Fill(std::move(energyTuple));
            });

        reconEnergy.Write();
    }

    if (reconstructTwoBody) {
        using ECALEnergy = Mustard::Data::TupleModel<
            Mustard::Data::Value<double, "Edep", "Energy deposition in total">,
            Mustard::Data::Value<double, "Edep1", "Energy deposition of the 1st cluster">,
            Mustard::Data::Value<muc::array3f, "Centroid1", "Centroid of the 1st cluster">,
            Mustard::Data::Value<double, "Edep2", "Energy deposition of the 2nd cluster">,
            Mustard::Data::Value<muc::array3f, "Centroid2", "Centroid of the 2nd cluster">,
            Mustard::Data::Value<double, "dE", "Energy difference of two reconstructed tracks">,
            Mustard::Data::Value<double, "dt", "Time difference of two reconstructed tracks">,
            Mustard::Data::Value<double, "cosTheta", "Angle between two reconstructed tracks">>;
        Mustard::Data::Output<ECALEnergy> reconEnergy{"G4Run0/ReconECAL"};

        Mustard::Data::Processor processor;
        processor.Process<Data::ECALSimHit>(
            ROOT::RDataFrame{cli->get("--input-tree"), cli->get<std::vector<std::string>>("input")}, int{}, "EvtID",
            [&](bool byPass, auto&& event) {
                if (byPass) {
                    return;
                }
                muc::timsort(event,
                             [](auto&& hit1, auto&& hit2) {
                                 return Get<"Edep">(*hit1) > Get<"Edep">(*hit2);
                             });

                std::unordered_map<int, std::shared_ptr<Mustard::Data::Tuple<Data::ECALSimHit>>> hitDict;
                std::vector<int> potentialSeedModule;

                for (auto&& hit : event) {
                    hitDict.try_emplace(Get<"ModID">(*hit), hit);

                    if (Get<"Edep">(*hit) < 15_MeV) {
                        continue;
                    }

                    potentialSeedModule.emplace_back(Get<"ModID">(*hit));
                }

                if (std::ssize(potentialSeedModule) < 2) {
                    return;
                }

                std::unordered_set<int> firstCluster;
                std::unordered_set<int> secondCluster;
                CLHEP::Hep3Vector firstClusterCentroid{};
                CLHEP::Hep3Vector secondClusterCentroid{};

                auto firstSeedModule = potentialSeedModule.begin();
                auto secondSeedModule = std::ranges::find_if(
                    potentialSeedModule,
                    [&](int m) {
                        const auto& c1{moduleList.at(*firstSeedModule).centroid};
                        const auto& c2{moduleList.at(m).centroid};
                        return c1.angle(c2) > 0.8 * pi;
                    });

                if (secondSeedModule == potentialSeedModule.end()) {
                    return;
                }

                const auto clustering = [&](std::unordered_set<int>& set,
                                            CLHEP::Hep3Vector& c,
                                            std::vector<int>::iterator seedIt) -> double {
                    const auto addClusterLayers = [&](int module) {
                        set.insert(module);
                        for (auto&& neighbor : moduleList.at(module).neighborModuleID) {
                            set.insert(neighbor);
                            for (auto&& secondNeighbor : moduleList.at(neighbor).neighborModuleID) {
                                set.insert(secondNeighbor);
                            }
                        }
                    };

                    addClusterLayers(*seedIt);

                    double totalEnergy{};
                    CLHEP::Hep3Vector weightedCentroid{};

                    for (const auto& module : set) {
                        auto hitIt = hitDict.find(module);

                        if (hitIt == hitDict.end() or Get<"Edep">(*hitIt->second) < 50_keV) {
                            continue;
                        }

                        auto energy = Get<"Edep">(*hitIt->second);
                        weightedCentroid += energy * moduleList.at(module).centroid;
                        totalEnergy += energy;
                    }
                    c = weightedCentroid / totalEnergy;

                    return totalEnergy;
                };

                auto firstClusterSignal = clustering(firstCluster, firstClusterCentroid, firstSeedModule);
                auto secondClusterSignal = clustering(secondCluster, secondClusterCentroid, secondSeedModule);

                Mustard::Data::Tuple<ECALEnergy> energyTuple;
                // total signal
                Get<"Edep">(energyTuple) = firstClusterSignal + secondClusterSignal;
                // the first cluster signal
                Get<"Edep1">(energyTuple) = firstClusterSignal;
                Get<"Centroid1">(energyTuple) = firstClusterCentroid;
                // the second cluster signal
                Get<"Edep2">(energyTuple) = secondClusterSignal;
                Get<"Centroid2">(energyTuple) = secondClusterCentroid;
                // signal differences
                Get<"dE">(energyTuple) = std::abs(firstClusterSignal - secondClusterSignal);
                Get<"dt">(energyTuple) = std::abs(*Get<"t">(*hitDict.at(*firstSeedModule)) - *Get<"t">(*hitDict.at(*secondSeedModule)));
                Get<"cosTheta">(energyTuple) = firstClusterCentroid.cosTheta(secondClusterCentroid);

                reconEnergy.Fill(std::move(energyTuple));
            });

        reconEnergy.Write();
    }

    if (reconstructInvisible) {
        using ECALEnergy = Mustard::Data::TupleModel<
            Mustard::Data::Value<double, "Edep", "Energy deposition in total">,
            Mustard::Data::Value<muc::array3f, "Centroid", "Centroid of the cluster">,
            Mustard::Data::Value<double, "t", "Time of the reconstructed tracks">,
            Mustard::Data::Value<double, "theta", "Angle between inital momentum direction">>;
        Mustard::Data::Output<ECALEnergy> reconEnergy{"G4Run0/ReconECAL"};

        Mustard::Data::Processor processor;
        processor.Process<Data::ECALSimHit>(
            ROOT::RDataFrame{cli->get("--input-tree"), cli->get<std::vector<std::string>>("input")}, int{}, "EvtID",
            [&](bool byPass, auto&& event) {
                if (byPass) {
                    return;
                }
                muc::timsort(event,
                             [](auto&& hit1, auto&& hit2) {
                                 return Get<"Edep">(*hit1) > Get<"Edep">(*hit2);
                             });

                std::unordered_map<int, std::shared_ptr<Mustard::Data::Tuple<Data::ECALSimHit>>> hitDict;
                std::vector<int> potentialSeedModule;

                for (auto&& hit : event) {
                    hitDict.try_emplace(Get<"ModID">(*hit), hit);

                    if (Get<"Edep">(*hit) < 1_MeV) {
                        continue;
                    }

                    potentialSeedModule.emplace_back(Get<"ModID">(*hit));
                }

                if (std::ssize(potentialSeedModule) < 1) {
                    return;
                }

                std::unordered_set<int> cluster;
                CLHEP::Hep3Vector clusterCentroid{};
                auto seedModule = potentialSeedModule.begin();

                const auto clustering = [&](std::unordered_set<int>& set,
                                            CLHEP::Hep3Vector& c,
                                            std::vector<int>::iterator seedIt) -> double {
                    const auto addClusterLayers = [&](int module) {
                        set.insert(module);
                        for (auto&& neighbor : moduleList.at(module).neighborModuleID) {
                            set.insert(neighbor);
                            for (auto&& secondNeighbor : moduleList.at(neighbor).neighborModuleID) {
                                set.insert(secondNeighbor);
                            }
                        }
                    };
                    addClusterLayers(*seedIt);

                    double totalEnergy{};
                    CLHEP::Hep3Vector weightedCentroid{};

                    for (const auto& module : set) {
                        auto hitIt = hitDict.find(module);

                        if (hitIt == hitDict.end() or Get<"Edep">(*hitIt->second) < 50_keV) {
                            continue;
                        }

                        auto energy = Get<"Edep">(*hitIt->second);
                        weightedCentroid += energy * moduleList.at(module).centroid;
                        totalEnergy += energy;
                    }
                    c = weightedCentroid / totalEnergy;

                    return totalEnergy;
                };

                auto clusterSignal = clustering(cluster, clusterCentroid, seedModule);

                Mustard::Data::Tuple<ECALEnergy> energyTuple;
                // total signal
                Get<"Edep">(energyTuple) = clusterSignal;
                Get<"Centroid">(energyTuple) = clusterCentroid;
                Get<"t">(energyTuple) = *Get<"t">(*hitDict.at(*seedModule));
                Get<"theta">(energyTuple) = clusterCentroid.theta(CLHEP::Hep3Vector{0, 0, 1});

                reconEnergy.Fill(std::move(energyTuple));
            });

        reconEnergy.Write();
    }

    return EXIT_SUCCESS;
}

} // namespace MACE::PhaseI::ReconECAL
