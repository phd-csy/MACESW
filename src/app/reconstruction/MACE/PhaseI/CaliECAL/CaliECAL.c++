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
#include "MACE/Data/SimVertex.h++"
#include "MACE/Detector/Description/ECAL.h++"
#include "MACE/PhaseI/CaliECAL/CaliECAL.h++"
#include "MACE/PhaseI/Detector/Description/UsePhaseIDefault.h++"
#include "MACE/Reconstruction/ECALClustering/Clusterer.h++"

#include "Mustard/CLI/BasicCLI.h++"
#include "Mustard/Data/Output.h++"
#include "Mustard/Data/SeqProcessor.h++"
#include "Mustard/Data/Tuple.h++"
#include "Mustard/Detector/Description/DescriptionIO.h++"
#include "Mustard/Env/BasicEnv.h++"
#include "Mustard/IO/PrettyLog.h++"
#include "Mustard/Parallel/ProcessSpecificPath.h++"
#include "Mustard/Utility/LiteralUnit.h++"
#include "Mustard/Utility/MathConstant.h++"

#include "CLHEP/Vector/ThreeVector.h"

#include "ROOT/RDataFrame.hxx"
#include "TFile.h"
#include "TTree.h"

#include "fmt/format.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace MACE::PhaseI::CaliECAL {

CaliECAL::CaliECAL() :
    Subprogram{"CaliECAL", "Electromagnetic calorimeter (ECAL) event reconstruction for calibration in PhaseI."} {}

using namespace Mustard::LiteralUnit;
using namespace Mustard::MathConstant;
using namespace std::literals;

auto CaliECAL::Main(int argc, char* argv[]) const -> int {
    Mustard::CLI::BasicCLI<> cli;
    cli->add_argument("input").help("Input file path(s).").nargs(argparse::nargs_pattern::at_least_one);
    cli->add_argument("-t", "--input-tree").help("Input tree name.").default_value("G4Run0/ECALSimHit"s).required().nargs(1);
    cli->add_argument("-o", "--output").help("Output file path.").required().nargs(1);
    cli->add_argument("-m", "--output-mode").help("Output file creation mode.").default_value("RECREATE"s).required().nargs(1);
    cli->add_argument("-d", "--description").help("Description YAML file path.").nargs(1);
    cli->add_argument("-optics", "--optics").help("Use optical response.").flag();
    Mustard::Env::BasicEnv env{argc, argv, cli};

    if (const auto descriptionPath{cli->present("--description")}) {
        Mustard::Detector::Description::DescriptionIO::Import<MACE::Detector::Description::ECAL>(*descriptionPath);
    } else {
        Detector::Description::UsePhaseIDefault();
    }

    const auto& ecal{MACE::Detector::Description::ECAL::Instance()};
    const auto& moduleList{ecal.Array().moduleList};

    using ECALEnergy = Mustard::Data::TupleModel<
        Mustard::Data::Value<double, "Edep", "Energy deposition of the cluster">,
        Mustard::Data::Value<int, "PE", "Photoelectron counts of the cluster">,
        // Mustard::Data::Value<double, "t", "Time of the track">,
        Mustard::Data::Value<muc::array3f, "Position", "Position of the cluster">,
        Mustard::Data::Value<double, "cosTheta", "Cosine of angle between the tracks">,
        Mustard::Data::Value<double, "theta", "Angle between the tracks">>;

    Mustard::Data::Tuple<ECALEnergy> energyTuple;

    auto setEnergyTuple = [&](std::vector<int>& potentialSeedModule, std::unordered_map<int, std::shared_ptr<Mustard::Data::Tuple<Data::ECALSimHit>>>& hitDict, CLHEP::Hep3Vector truthHitMomentum) -> void {
        double energy{};
        int pe{};
        CLHEP::Hep3Vector weightedPosition{};
        CLHEP::Hep3Vector clusterPosition{};

        auto seedModule = potentialSeedModule.begin();
        auto cluster = ECALClustering::Clusterer(*seedModule, moduleList);
        for (const auto& module : cluster) {
            auto hitIt = hitDict.find(module);
            if (hitIt == hitDict.end() or Get<"Edep">(*hitIt->second) < 50_keV) {
                continue;
            }
            auto e = Get<"Edep">(*hitIt->second);
            weightedPosition += e * moduleList.at(module).centroid;
            energy += e;

            auto pe = Get<"nOptPho">(*hitIt->second);
            if (cli["--optics"] == true and pe > 3) {
                pe += pe;
            }
        }

        if (energy != 0) {
            clusterPosition = weightedPosition / energy;
        }

        Get<"Edep">(energyTuple) = energy;
        Get<"PE">(energyTuple) = pe;
        Get<"Position">(energyTuple) = clusterPosition;
        Get<"cosTheta">(energyTuple) = clusterPosition.cosTheta(truthHitMomentum);
    };

    auto primaryData = ROOT::RDataFrame{"G4Run0/SimPrimaryVertex", cli->get<std::vector<std::string>>("input")};
    auto inputData = ROOT::RDataFrame{cli->get("--input-tree"), cli->get<std::vector<std::string>>("input")};
    TFile outputFile{Mustard::Parallel::ProcessSpecificPath(cli->get("--output").c_str()).generic_string().c_str(), cli->get("--output-mode").c_str()};
    Mustard::Data::Output<ECALEnergy> reconEnergy{"G4Run0/CaliECAL"};
    Mustard::Data::SeqProcessor processor;

    processor.Process<Data::SimPrimaryVertex, Data::ECALSimHit>(
        {primaryData, inputData}, int{}, "EvtID",
        [&](auto&& primary, auto&& event) {
            muc::timsort(event,
                         [](auto&& hit1, auto&& hit2) {
                             return Get<"Edep">(*hit1) > Get<"Edep">(*hit2);
                         });

            std::unordered_map<int, std::shared_ptr<Mustard::Data::Tuple<Data::ECALSimHit>>> hitDict;
            std::vector<int> potentialSeedModule;
            muc::array3f primaryMomentum{};
            CLHEP::Hep3Vector truthHitMomentum{};

            for (auto&& hit : event) {
                hitDict.try_emplace(Get<"ModID">(*hit), hit);
                potentialSeedModule.emplace_back(Get<"ModID">(*hit));
            }

            if (primary.size() != 1) {
                Mustard::PrintError(fmt::format("Unexpected number ({}) of primary vertices", primary.size()));
                return;
            }
            primaryMomentum = Get<"p0">(*primary.front());
            truthHitMomentum.set(
                primaryMomentum.at(0),
                primaryMomentum.at(1),
                primaryMomentum.at(2));

            setEnergyTuple(potentialSeedModule, hitDict, truthHitMomentum);

            reconEnergy.Fill(std::move(energyTuple));
        });
    reconEnergy.Write();

    return EXIT_SUCCESS;
}

} // namespace MACE::PhaseI::CaliECAL
